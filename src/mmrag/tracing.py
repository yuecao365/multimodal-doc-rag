"""Optional tracing of model calls (OpenTelemetry via OpenInference, viewed in Phoenix).

Off unless ``enable`` is called (``--trace`` on the experiment runner). It needs
the ``tracing`` dependency group and a running Phoenix server:

    uv sync --group tracing
    uvx arize-phoenix serve          # UI and collector on http://localhost:6006

Every OpenAI call then becomes a span; ``span`` groups the calls of one unit of
work (one agent question) into one trace.
"""

from __future__ import annotations

from contextlib import contextmanager

_tracer = None


def enable(endpoint: str = "http://localhost:6006/v1/traces", project: str = "mmrag") -> None:
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    global _tracer
    provider = TracerProvider(resource=Resource({"openinference.project.name": project}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint)))
    trace.set_tracer_provider(provider)
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    _tracer = trace.get_tracer("mmrag")


@contextmanager
def span(name: str, **attributes: str):
    """A parent span around a block; does nothing when tracing is off."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name, attributes={"openinference.span.kind": "CHAIN"}) as s:
        for key, value in attributes.items():
            s.set_attribute("input.value" if key == "question" else key, value)
        yield
