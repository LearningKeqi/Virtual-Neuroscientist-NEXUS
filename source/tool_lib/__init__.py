from importlib import import_module
from pkgutil import iter_modules
from langchain.tools.base import BaseTool

def _discover_tools() -> list[BaseTool]:
    tool_objs: list[BaseTool] = []
    
    for info in iter_modules(__path__):
        mod = import_module(f"{__name__}.{info.name}")
        for obj in vars(mod).values():
            if isinstance(obj, BaseTool):
                tool_objs.append(obj)
    return tool_objs

# all_tools: list[BaseTool] = _discover_tools()
# __all__ = ["all_tools"]


def _discover_qc_tools() -> list[BaseTool]:
    try:
        qc_module = import_module(f"{__name__}.quality_control")
    except ModuleNotFoundError:
        return []
    qc_tool_objs: list[BaseTool] = [
        obj for obj in vars(qc_module).values() if isinstance(obj, BaseTool)
    ]
    return qc_tool_objs

def _discover_data_awareness_tools() -> list[BaseTool]:
    try:
        data_awareness_module = import_module(f"{__name__}.data_awareness")
    except ModuleNotFoundError:
        return []
    data_awareness_tool_objs: list[BaseTool] = [
        obj for obj in vars(data_awareness_module).values() if isinstance(obj, BaseTool)
    ]
    return data_awareness_tool_objs


def _discover_downstream_analysis_tools() -> list[BaseTool]:
    try:
        downstream_analysis_module = import_module(f"{__name__}.downstream_analysis")
    except ModuleNotFoundError:
        return []
    downstream_analysis_tool_objs: list[BaseTool] = [
        obj for obj in vars(downstream_analysis_module).values() if isinstance(obj, BaseTool)
    ]
    return downstream_analysis_tool_objs



def _discover_processing_execution_tools() -> list[BaseTool]:
    try:
        processing_execution_module = import_module(f"{__name__}.processing_execution")
    except ModuleNotFoundError:
        return []
    processing_execution_tool_objs: list[BaseTool] = [
        obj for obj in vars(processing_execution_module).values() if isinstance(obj, BaseTool)
    ]
    return processing_execution_tool_objs




qc_tools: list[BaseTool] = _discover_qc_tools()
data_awareness_tools: list[BaseTool] = _discover_data_awareness_tools()
downstream_analysis_tools: list[BaseTool] = _discover_downstream_analysis_tools()
processing_execution_tools: list[BaseTool] = _discover_processing_execution_tools()


__all__ = ["qc_tools", "data_awareness_tools", "downstream_analysis_tools", "processing_execution_tools"]


