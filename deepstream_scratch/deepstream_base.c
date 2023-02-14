#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"

/* Apparently this limits the number of characters displayed on the OSD */
#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0 // class "V" //
#define PGIE_CLASS_ID_PERSON 2 // class "P" //

/* Apparently we have to define resolution for the muxer to scale all sources. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* A timeout. Cuz why tf not. Also microseconds? */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* Check for parsing error. Urgh, define ****. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
    if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
        g_printerr("Configuration file is fucked.\n"); \
        return -1; \
    }

gint frame_number = 0;
gchar pgie_classes_str[4][32] = {"V", "TW", "P", "R"};

/* Display stuffs on the on-screen display. */
static GstPadProbeReturn osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    /* Frame something something for loop */
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        /* ANOTHER FOR LOOP FUCKKKKKKKKKKKKKK */
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);
                
        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    g_print ("Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_number, num_rects, vehicle_count, person_count);
    frame_number++;
    return GST_PAD_PROBE_OK;
}

/* Handling main event loop something idk, probably something the device runs. */
static gboolean bus_call (GstBus * bus, GstMessage * msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_EOS: {
            g_print ("End of stream. Now fuck off.");
            g_main_loop_quit (loop);
            break;
        }
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error (msg, &error, &debug);
            g_printerr ("ERROR from element %s: %s\n",
                GST_OBJECT_NAME (msg->src), error->message);
            if (debug) { g_printerr ("Error details: %s\n", debug); }
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

// glib types are used to ensure compatibility
// idk

int main (int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
        *decoder = NULL, *streammux = NULL, *sink = NULL,
        *pgie = NULL, *nvvidconv = NULL, *nvosd = NULL; /* PGIE: Plugin (?) GPU Inference Engine */
    
    GstBus *bus = NULL;
    guint bus_watch_id;
    
    GstPad *osd_sink_pad = NULL;
    gboolean yaml_config = false;
    NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER; /* this enables the use of TensorRT inference*/

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);

    yaml_config = (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml"));
    if (yaml_config) {
        RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
                        "primary-gie"));
    }
    g_print ("Got here.\n");

    pipeline = gst_pipeline_new ("dsbase-pipeline");
    
    source = gst_element_factory_make ("filesrc", "file-source");
    
    h264parser = gst_element_factory_make ("h264parse", "h264-parser");
    decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");
    
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr ("Pipeline or mux died. Exiting.\n");
        return -1;
    }

    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

    /* No idea what this is, will return later. Can replace with fakesink if no osd is used.*/
    if(prop.integrated) {
        sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
    } else {
        sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
    }

    if (!source || !h264parser || !decoder || !pgie
        || !nvvidconv || !nvosd || !sink) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }
    g_object_set (G_OBJECT (source), "location", argv[1], NULL);
    
    /* if a .h264 file is passed in instead */
    if (g_str_has_suffix (argv[1], ".h264")) {
        g_object_set (G_OBJECT (source), "location", argv[1], NULL);
        g_object_set (G_OBJECT (streammux), "batch-size", 1, NULL);
        g_object_set (G_OBJECT (streammux),
                    "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT,
                    "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
        
        g_object_set (G_OBJECT (pgie),
                    "config-file-path", "dsbase-config.txt", NULL);
    }

    if (yaml_config) {
        RETURN_ON_PARSER_ERROR(nvds_parse_file_source(source, argv[1], "source"));
        RETURN_ON_PARSER_ERROR(nvds_parse_streammux(streammux, argv[1], "streammux"));
        RETURN_ON_PARSER_ERROR(nvds_parse_gie(pgie, argv[1], "primary-gie"));
    }

    /* Message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    /* Pipeline, assemble. */
    gst_bin_add_many (GST_BIN (pipeline),
        source, h264parser, decoder, streammux, pgie,
        nvvidconv, nvosd, sink, NULL);
    g_print ("Elements added to bin. AAAAAAAA\n");

    /* Adding pads */
    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";
    
    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    if (!sinkpad) {
        g_printerr ("Streammux request sink pad failed. Exiting.\n");
        return -1;
    }

    srcpad = gst_element_get_static_pad (decoder, pad_name_src);
    if (!srcpad) {
        g_printerr ("Decoder request src pad failed. Exiting.\n");
        return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);

    if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
        g_printerr ("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }

    if (!gst_element_link_many (streammux, pgie,
            nvvidconv, nvosd, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
    }

    osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
    
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
            osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref (osd_sink_pad);

    /* Set the pipeline to "playing" state */
    g_print ("Using file: %s\n", argv[1]);
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");
    g_main_loop_run (loop);

    g_print ("Ended. AHHHHHHHHH\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline.\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    return 0;
    /* Remember to add error checking */
}