import sys

def fix_all():
    with open('core/lifecycle_orchestrator.py', 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace('import logging\nimport time', 'import logging\nimport time\nimport warnings')

    import re
    
    # 1. Lifecycle docstring 1
    content = re.sub(
        r'"""Phase 1: drive per-file lifecycles; Phase 2: run the global DAG\.\n\n        Phase 1 completes when.*?\n        delegates graph execution to :meth:~AgentManager\.execute_graph\.\n        """',
        '"""Phase 1: drive per-file lifecycles; Phase 2: run the global DAG.\n\n        .. deprecated:: Use PipelineExecutor.\n        """\n        warnings.warn("LifecycleOrchestrator.execute_with_lifecycle is deprecated. Use PipelineExecutor directly.", DeprecationWarning, stacklevel=2)',
        content,
        flags=re.DOTALL
    )

    # 2. Lifecycle docstring 2
    content = re.sub(
        r'"""Tier-scheduled execution with repo-level build checkpoints\.\n\n        Delegates entirely to :class:~core\.pipeline_executor\.PipelineExecutor\n        which owns the tier-blocking and checkpoint-retry logic\.\n        """',
        '"""Tier-scheduled execution with repo-level build checkpoints.\n\n        .. deprecated:: Use PipelineExecutor.\n        """\n        warnings.warn("LifecycleOrchestrator.execute_with_checkpoints is deprecated. Use PipelineExecutor directly.", DeprecationWarning, stacklevel=2)',
        content,
        flags=re.DOTALL
    )
    
    with open('core/lifecycle_orchestrator.py', 'w', encoding='utf-8') as f:
        f.write(content)

    with open('core/agent_manager.py', 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace('import logging\nfrom typing', 'import logging\nimport warnings\nfrom typing')

    # 3. AgentManager docstring 1
    content = re.sub(
        r'"""Drive per-file lifecycle FSM then run the global DAG\.\n\n        Delegates to :class:~core\.lifecycle_orchestrator\.LifecycleOrchestrator\.\n        """',
        '"""Drive per-file lifecycle FSM then run the global DAG.\n\n        .. deprecated:: Use PipelineExecutor.\n        """\n        warnings.warn("AgentManager.execute_with_lifecycle is deprecated. Use PipelineExecutor directly.", DeprecationWarning, stacklevel=2)',
        content,
        flags=re.DOTALL
    )

    # 4. AgentManager docstring 2
    content = re.sub(
        r'"""Tier-scheduled execution with repo-level build checkpoints\.\n\n        Delegates to :class:~core\.lifecycle_orchestrator\.LifecycleOrchestrator\.\n        """',
        '"""Tier-scheduled execution with repo-level build checkpoints.\n\n        .. deprecated:: Use PipelineExecutor.\n        """\n        warnings.warn("AgentManager.execute_with_checkpoints is deprecated. Use PipelineExecutor directly.", DeprecationWarning, stacklevel=2)',
        content,
        flags=re.DOTALL
    )

    with open('core/agent_manager.py', 'w', encoding='utf-8') as f:
        f.write(content)

fix_all()
print("Done.")
