FROM ffreis/base-builder

USER root

RUN mkdir -p /build \
    && chown appuser:appgroup /build \
    && chmod 0750 /build

WORKDIR /build

USER appuser:appgroup

COPY --chown=appuser:appgroup app/ .

USER appuser:appgroup

RUN python3 -m pip install --user -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["main.py"]
