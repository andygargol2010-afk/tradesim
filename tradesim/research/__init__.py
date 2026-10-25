"""Market research for agents: news (with Claude analysis), macro data by asset type, the event calendar
of the coming month, and technical patterns. See :func:`tradesim.research.refresh.refresh_research`."""

from tradesim.research.calendar import CalendarBuilder, Event
from tradesim.research.macro import MacroMonitor, relevant_indicators
from tradesim.research.news import NewsCollector, NewsItem, NewsStore
from tradesim.research.refresh import ResearchConfig, ResearchStore, refresh_research

__all__ = ["CalendarBuilder", "Event", "MacroMonitor", "relevant_indicators", "NewsCollector", "NewsItem",
           "NewsStore", "ResearchConfig", "ResearchStore", "refresh_research"]
