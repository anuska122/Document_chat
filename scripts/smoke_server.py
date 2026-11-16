"""Compatibility entry point for the real local application.

The former test-response preview was removed. This starts the real local server,
including the configured LLM, persistent files and real embedding retrieval.
"""
if __name__ == "__main__":
    import runpy

    runpy.run_module("scripts.local_server", run_name="__main__")
