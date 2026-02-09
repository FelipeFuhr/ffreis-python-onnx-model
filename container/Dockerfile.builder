FROM ffreis/base-builder

USER root

RUN mkdir -p /build \
    && chown appuser:appgroup /build \
    && chmod 0750 /build

WORKDIR /build

USER appuser:appgroup

COPY --chown=appuser:appgroup app/ .

# No dependencies to install for now (requirements.txt is just a placeholder)

ENTRYPOINT ["python3"]
CMD ["main.py"]
