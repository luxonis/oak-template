import os
from pathlib import Path

import depthai as dai
from utils.snaps_producer import SnapsProducer
from depthai_nodes.node.parsing_neural_network import ParsingNeuralNetwork


visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device()

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")
    platform = device.getPlatform()
    model_description = dai.NNModelDescription.fromYamlFile(
        f"yolov6_nano_r2_coco.{platform.name}.yaml"
    )
    nn_archive = dai.NNArchive(dai.getModelFromZoo(modelDescription=model_description))

    input_node = pipeline.create(dai.node.Camera).build()

    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(input_node, nn_archive)

    visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
    visualizer.addTopic("Visualizations", nn_with_parser.out, "images")

    snaps_producer = pipeline.create(SnapsProducer).build(
        nn_with_parser.passthrough,
        nn_with_parser.out,
        label_map=nn_archive.getConfigV1().model.heads[0].metadata.classes,
    )

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
