
# components/__init__.py
"""Components package for Risk Killer application"""

# 각 탭 모듈들을 import할 수 있도록 설정
try:
    from .tab_regulation import show_regulation_chat
    from .tab_recall import show_recall_chat
    from .tab_news import show_news
    from .tab_tableau import create_market_dashboard
    from .tab_export import show_export_helper
    
    __all__ = [
        'show_regulation_chat',
        'show_recall_chat', 
        'show_news',
        'create_market_dashboard',
        'show_export_helper'
    ]
except ImportError as e:
    print(f"Components import warning: {e}")
    pass