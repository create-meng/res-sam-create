"""
全局日志系统
统一管理所有实验脚本的日志输出
"""
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

# 全局日志配置
LOG_DIR = None
GLOBAL_LOGGER = None


def setup_global_logger(
    base_dir: str,
    script_name: str,
    log_level: int = logging.DEBUG
) -> logging.Logger:
    """
    设置全局日志记录器
    
    Parameters:
    -----------
    base_dir : str
        项目根目录
    script_name : str
        脚本名称（如 "01_build_feature_bank_v17", "02_inference_auto_v17"）
    log_level : int
        日志级别
        
    Returns:
    --------
    logging.Logger
        配置好的日志记录器
    """
    global LOG_DIR, GLOBAL_LOGGER
    
    # 创建日志目录
    LOG_DIR = os.path.join(base_dir, "outputs", "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 日志文件名：v17_YYYYMMDD.log（同一天的所有操作记录在同一个文件）
    log_filename = f"v17_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(LOG_DIR, log_filename)
    
    # 创建logger
    logger = logging.getLogger("res_sam_v17")
    logger.setLevel(log_level)
    
    # 清除已有的handlers（避免重复）
    logger.handlers.clear()
    
    # 文件handler - 详细日志
    fh = RotatingFileHandler(
        log_path,
        maxBytes=100*1024*1024,  # 100MB
        backupCount=10,
        encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)
    
    # 控制台handler - 只显示WARNING及以上级别（简洁输出）
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    
    # 格式化
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | [%(script)s] | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(message)s'
    )
    
    fh.setFormatter(file_formatter)
    ch.setFormatter(console_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # 添加script_name到所有日志记录
    class ScriptFilter(logging.Filter):
        def filter(self, record):
            record.script = script_name
            return True
    
    logger.addFilter(ScriptFilter())
    
    GLOBAL_LOGGER = logger
    
    # 记录脚本开始（只在日志文件）
    logger.info("=" * 80)
    logger.info(f"{script_name} 开始执行")
    logger.info(f"日志文件: {log_path}")
    logger.info("=" * 80)
    
    return logger


def get_logger() -> Optional[logging.Logger]:
    """获取全局日志记录器"""
    return GLOBAL_LOGGER


def log_config(config: dict, logger: Optional[logging.Logger] = None):
    """
    记录配置参数
    
    Parameters:
    -----------
    config : dict
        配置字典
    logger : logging.Logger, optional
        日志记录器，如果为None则使用全局logger
    """
    if logger is None:
        logger = get_logger()
    
    if logger is None:
        return
    
    logger.info("配置参数:")
    for key, value in sorted(config.items()):
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        elif isinstance(value, (list, tuple)) and len(value) > 10:
            logger.info(f"  {key}: [{len(value)} items]")
        else:
            logger.info(f"  {key}: {value}")


def log_section(title: str, logger: Optional[logging.Logger] = None):
    """
    记录章节标题
    
    Parameters:
    -----------
    title : str
        章节标题
    logger : logging.Logger, optional
        日志记录器
    """
    if logger is None:
        logger = get_logger()
    
    if logger is None:
        return
    
    logger.info("-" * 80)
    logger.info(title)
    logger.info("-" * 80)


def log_step(step_name: str, details: dict = None, logger: Optional[logging.Logger] = None):
    """
    记录处理步骤
    
    Parameters:
    -----------
    step_name : str
        步骤名称
    details : dict, optional
        详细信息
    logger : logging.Logger, optional
        日志记录器
    """
    if logger is None:
        logger = get_logger()
    
    if logger is None:
        return
    
    logger.info(f"[{step_name}]")
    if details:
        for key, value in details.items():
            logger.info(f"  {key}: {value}")


def log_finish(script_name: str, logger: Optional[logging.Logger] = None):
    """
    记录脚本结束
    
    Parameters:
    -----------
    script_name : str
        脚本名称
    logger : logging.Logger, optional
        日志记录器
    """
    if logger is None:
        logger = get_logger()
    
    if logger is None:
        return
    
    logger.info("=" * 80)
    logger.info(f"{script_name} 执行完成")
    logger.info("=" * 80)
    logger.info("")  # 空行分隔
