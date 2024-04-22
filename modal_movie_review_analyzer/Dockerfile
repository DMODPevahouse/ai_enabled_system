FROM python:3.9
COPY . ./
RUN pip3 install -r requirements.txt
RUN python -m spacy download en_core_web_sm
CMD ["modal_service.py"]
ENTRYPOINT ["python"]