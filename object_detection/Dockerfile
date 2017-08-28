FROM registry.w3ibm.bluemix.net/holmes/obj_base

RUN cd /home && mkdir object_detection && cd object_detection && mkdir protos && mkdir data && mkdir utils \
 && mkdir ssd_mobilenet_v1_coco_11_06_2017

COPY . /home/object_detection/
COPY protos /home/object_detection/protos/
COPY data /home/object_detection/data/
COPY ssd_mobilenet_v1_coco_11_06_2017 /home/object_detection/ssd_mobilenet_v1_coco_11_06_2017/
COPY utils /home/object_detection/utils/

RUN chmod -R 775 /home/object_detection/

EXPOSE 8080

WORKDIR /home/object_detection

ENTRYPOINT ["python","/home/object_detection/start.py"]
