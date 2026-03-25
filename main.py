import depthai as dai
import os

from depthai_nodes.node import SnapsUploader
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork
from utils.snaps_producer import SnapsProducer
from dotenv import load_dotenv

# Assumes `DEPTHAI_HUB_API_KEY` is defined in the workspace root `.env` file.
# Load environment variables before initializing the pipeline.
load_dotenv(override=True)  

model = "luxonis/yolov6-nano:r2-coco-512x288"
time_interval = 10.0  # min nr of seconds between snaps uploading

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    model_description = dai.NNModelDescription(model)
    platform = device.getPlatformAsString()
    model_description.platform = platform
    nn_archive = dai.NNArchive(
        dai.getModelFromZoo(
            model_description,
        )
    )

    input_node = pipeline.create(dai.node.Camera).build()

    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        input_node, nn_archive
    )

    visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
    visualizer.addTopic("Visualizations", nn_with_parser.out, "images")

    snaps_producer = pipeline.create(SnapsProducer).build(
        frame=nn_with_parser.passthrough,
        detections=nn_with_parser.out,
        time_interval=time_interval
    )

    snaps_uploader = pipeline.create(SnapsUploader).build(snaps_producer.out)

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
