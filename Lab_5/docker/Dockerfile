FROM python:3.8.1
COPY ./*.py /exp/
# COPY ./serve.py /exp/
# COPY ./models/*.joblib /exp/
COPY ./requirements.txt /exp/requirements.txt
# RUN pip3 install -U scikit-learn
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD ["python", "./M22CS061_Lab_Assignment_5.py"]
# CMD python plot_final.py --clf_model ${CLF_MODEL} --random_state ${R_S}
# docker run -p 5000:5000 -it exp:v1
# docker build -t dock_mlp:v1 -f docker/Dockerfile . --network=host