/* 
Feb 16, 2023
Deepstream Pipeline and frame for further developments
Written in C++ because why not?
All configurations will be obtained through .yaml files
*/
#include <iostream>
#include <string.h>
#include <cmath>

#include <gst/gst.h>
#include <glib.h>
#include <cuda_runtime_api.h>

#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"

static gboolean PERF_MODE = FALSE;

bool check_yaml_parser (gboolean check_result) {
    if (NVDS_YAML_PARSER_SUCCESS != check_result) {
        std::cerr << ".yaml file is buggy. Check again.\n";
        return false;
    }
    return true;
}

int get_device_type () {
    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    return prop.integrated;
}

static void cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data) {
    gchar *feature_NVIDIA = "memory:NVMM";
    GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
    if (!caps) {
        caps = gst_pad_query_caps (decoder_src_pad, NULL);
    }
    const GstStructure *str = gst_caps_get_structure (caps, 0);
    const gchar *name = gst_structure_get_name (str);
    GstElement *source_bin = (GstElement *) data;
    GstCapsFeatures *features = gst_caps_get_features (caps, 0);

    if (!strncmp (name, "video", 5)) {
        if (gst_caps_features_contains (features, feature_NVIDIA)) {
            GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
            if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad), decoder_src_pad)) {
                std::cerr << "Failed to link decoder src pad to source bin ghost pad.\n";
            }

        } else {
            std::cerr << "Nvidia Decode Plugin not used.\n";
        }
    }

}

static void decodebin_child_added (GstChildProxy * child_proxy, GObject * object, gchar * name, gpointer user_data) {
    std::cout << "Decodebin child added: " << name << "\n" ;
    if (g_strrstr (name, "decodebin") == name) {
        g_signal_connect (G_OBJECT (object), "child-added", G_CALLBACK (decodebin_child_added), user_data);
    }
    if (g_strrstr (name, "source") == name) {
        g_object_set (G_OBJECT (object), "drop-on-latency", TRUE, NULL);
    }
}

static GstElement * create_source_bin (guint index, gchar * uri) {
    GstElement *bin = NULL, *uri_decode_bin = NULL;
    gchar bin_name[16] = { };

    g_snprintf (bin_name, 15, "source-bin-%02d", index);
    std::cout << "Bin name: " << bin_name << "\n";

    bin = gst_bin_new (bin_name);
    if (PERF_MODE) {
        uri_decode_bin = gst_element_factory_make ("nvurisrcbin", "uri-decode-bin");
        g_object_set (G_OBJECT (uri_decode_bin), "cudadec-memtype", 0, NULL);
    } else {
        uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");
    }

    if (!bin || !uri_decode_bin) {
        std::cerr << "Element creation failed.\n";
        return NULL;
    }

    g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

    g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added", G_CALLBACK (cb_newpad), bin);
    g_signal_connect (G_OBJECT (uri_decode_bin), "child-added", G_CALLBACK (decodebin_child_added), bin);

    gst_bin_add (GST_BIN (bin), uri_decode_bin);

    if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src", GST_PAD_SRC))) {
        std::cerr << "Failed to add ghost pad in source bin.\n";
        return NULL;
    }

    return bin;
}

static gboolean bus_call (GstBus * bus, GstMessage * msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_EOS: {
            std::cout << "End of stream.\n";
            g_main_loop_quit (loop);
            break;
        }
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error (msg, &error, &debug);
            std::cerr << "Error from " << GST_OBJECT_NAME (msg->src) << error->message << "\n";
            if (debug) { std::cerr << "Error details: " << debug << "\n"; }
            g_free (debug);
            g_error_free (error);
            g_main_loop_quit (loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

static GstPadProbeReturn tiler_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == 0) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == 2) {
                person_count++;
                num_rects++;
            }
        }
        g_print ("Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_meta->frame_num, num_rects, vehicle_count, person_count);
    }
    return GST_PAD_PROBE_OK;
}

int main(int argc, char *argv[]) {
    std::cout << "Begin program.\n";
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
        *decoder = NULL, *streammux = NULL, *sink = NULL,
        *tiler = NULL, 
        *pgie = NULL, *nvvidconv = NULL,    /* PGIE: Plugin (?) GPU Inference Engine */
        *nvosd = NULL, *nvdslogger = NULL;
    GstElement *queue1, *queue2, *queue3, *queue4, *queue5;
    
    GstBus *bus = NULL;
    guint bus_watch_id;
    
    guint pgie_batch_size;
    
    GstPad *tiler_src_pad = NULL;
    guint tiler_rows, tiler_columns;

    gboolean yaml_config = false;
    guint i = 0, num_sources = 0;

    NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER; /* this enables the use of TensorRT inference.
                                                      Change to _INFER_SERVER for Triton backend. */

    gchar *env_perf_mode = "NVDS_FRAME_PERF_MODE";
    PERF_MODE = g_getenv(env_perf_mode) && !g_strcmp0(g_getenv(env_perf_mode), "1");
    
    /* ---------------------------------------------------------------------------------------------
                                            Program starts here
    ------------------------------------------------------------------------------------------------ */
    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);

    yaml_config = (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml"));
    
    if (!yaml_config) {
        std::cerr << "User must specify .yml or .yaml configuration file.\n";
        return -1;
    }

    if (check_yaml_parser(nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie"))) {
        std::cout << "Specified gie.\n";
    } else {
        std::cerr << "No gie found.\nExiting.\n";
        return -1;
    }

    pipeline = gst_pipeline_new ("main-pipeline");
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
    
    if (!pipeline || !streammux) {
        std::cerr << "Element creation failed.\n";
    }

    gst_bin_add (GST_BIN (pipeline), streammux);

    GList *src_list = NULL;
    
    if (check_yaml_parser(nvds_parse_source_list(&src_list, argv[1], "source-list"))) {
        GList * temp = src_list;
        while (temp) {
            num_sources++;
            temp = temp->next;
        }
        g_list_free (temp);
    }
    std::cout << "Number of files: " << num_sources << "\n";

    for (i = 0; i < num_sources; i++) {
        GstPad *srcpad, *sinkpad;
        gchar pad_name[16] = { };

        GstElement *source_bin = NULL;
        std::cout << "Now playing: " << (char*) (src_list)->data << "\n";
        source_bin = create_source_bin (i, (char*) (src_list)->data);

        if (!source_bin) {
            std::cerr << "Failed to create source bin.\n";
            return -1;
        }

        gst_bin_add (GST_BIN (pipeline), source_bin);
        
        g_snprintf (pad_name, 15, "sink_%u", i);
        std::cout << "Pad name: " << pad_name << "\n";
        sinkpad = gst_element_get_request_pad (streammux, pad_name);

        if (!sinkpad) {
            std::cerr << "Streammux request sinkpad failed.\n";
            return -1;
        }

        srcpad = gst_element_get_static_pad (source_bin, "src");

        if (!srcpad) {
            std::cerr << "Failed to get src pad of source bin.\n";
            return -1;
        }

        if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
            std:: cerr << "Failed to link source bin to stream muxer.\n";
            return -1;
        }

        gst_object_unref (srcpad);
        gst_object_unref (sinkpad);

        src_list = src_list->next;
    }

    g_list_free (src_list);

    if (pgie_type == NVDS_GIE_PLUGIN_INFER) {
        pgie = gst_element_factory_make ("nvinfer", "primary-inference-engine");
    } else {
        pgie = gst_element_factory_make ("nvinferserver", "primary-inference-engine");
    }

    queue1 = gst_element_factory_make ("queue", "queue1");
    queue2 = gst_element_factory_make ("queue", "queue2");
    queue3 = gst_element_factory_make ("queue", "queue3");
    queue4 = gst_element_factory_make ("queue", "queue4");
    queue5 = gst_element_factory_make ("queue", "queue5");

    nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");
    tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

    if (PERF_MODE) {
        sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
    } else {
        if (get_device_type()) {
            sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
        } else {
            sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
        }
    }

    if (!pgie || !nvdslogger || !tiler || !nvvidconv || !nvosd || !sink) {
        std::cerr << "One element could not be created. Exiting.\n";
        return -1;
    }
    
    if (!check_yaml_parser (nvds_parse_streammux (streammux, argv[1], "streammux"))) {
        std::cerr << "Something is wrong with the streammux configuration.\n";
        return -1;
    }
    if (!check_yaml_parser (nvds_parse_gie (pgie, argv[1], "primary-gie"))) {
        std::cerr << "Something is wrong with the primary GIE configuration.\n";
        return -1;
    }

    g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != num_sources) {
        std::cerr << "WARNING: Imbalance batch-size from configuration vs number of sources. Overriding.\n";
        g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
    }

    if (!check_yaml_parser (nvds_parse_osd (nvosd, argv[1], "osd"))) {
        std::cerr << "Something is wrong with the on-screen-display configuration.\n";
        return -1;
    }

    tiler_rows = (guint) sqrt (num_sources);
    tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
    
    g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);
    if (!check_yaml_parser (nvds_parse_tiler (tiler, argv[1], "tiler"))) {
        std::cerr << "Something is wrong with the tiler configuration.\n";
        return -1;
    }
    if (!PERF_MODE) {
        if (get_device_type()) {
            if (!check_yaml_parser (nvds_parse_3d_sink (sink, argv[1], "sink"))) {
                std::cerr << "Something is wrong with the 3D configuration.\n";
                return -1;
            }
        } else {
            if (!check_yaml_parser (nvds_parse_egl_sink (sink, argv[1], "sink"))) {
                std::cerr << "Something is wrong with the EGL configuration.\n";
                return -1;
            }
        }
    }

    if (PERF_MODE) {
        if(get_device_type()) {
            g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 4, NULL);
        } else {
            g_object_set (G_OBJECT (streammux), "nvbuf-memory-type", 2, NULL);
        }
    }

    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2, nvdslogger, tiler,
                    queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL);
    if (!gst_element_link_many (streammux, queue1, pgie, queue2, nvdslogger, tiler,
        queue3, nvvidconv, queue4, nvosd, queue5, sink, NULL)) {
        std::cerr << "Elements could not be linked. Exiting.\n";
        return -1;
    }

    tiler_src_pad = gst_element_get_static_pad (pgie, "src");
    if (!tiler_src_pad) {
        std::cerr << "Unable to get src pad.\n";
    } else {
        gst_pad_add_probe (tiler_src_pad, GST_PAD_PROBE_TYPE_BUFFER, tiler_src_pad_buffer_probe, NULL, NULL);
    }
    gst_object_unref (tiler_src_pad);

    std::cout << "Using file: " << argv[1] << "\n";

    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    g_print ("Running...\n");
    g_main_loop_run (loop);

    g_print ("Returned, stopping playback...\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline eh.\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    return 0;
}
