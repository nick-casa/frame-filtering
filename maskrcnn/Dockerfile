FROM pytorch/torchserve:latest

COPY ["config.properties", "model.py", "index_to_name.json", "/home/model-server/"]

USER root
RUN apt-get update && apt-get install -y curl
RUN curl https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth -o /home/model-server/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth

RUN touch /home/model-server/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
RUN touch /home/model-server/model.py
RUN touch /home/model-server/config.properties
RUN touch /home/model-server/index_to_name.json
RUN mkdir -p /home/model-server/model-store
RUN ls -l /home/model-server/

RUN torch-model-archiver \
    --model-name maskrcnn \
    --version 1.0 \
    --model-file /home/model-server/model.py \
    --serialized-file /home/model-server/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth \
    --handler object_detector \
    --extra-files /home/model-server/index_to_name.json \
    --export-path=/home/model-server/model-store


RUN rm /home/model-server/model.py /home/model-server/index_to_name.json  /home/model-server/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth

CMD ["torchserve", \
    "--start", \
    "--model-store", "model-store", \
    "--ts-config", "config.properties", \
    "--models", "maskrcnn=maskrcnn.mar"]
