"""NYSE trading-day helpers shared by the signal and volatility modules."""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday,
                                    USLaborDay, USMartinLutherKingJr, USMemorialDay,
                                    USPresidentsDay, USThanksgivingDay,
                                    nearest_workday, sunday_to_monday)


class NYSEHolidays(AbstractHolidayCalendar):
    """NYSE full-closure holidays - unlike the US federal calendar, includes
    Good Friday and excludes Columbus/Veterans Day (NYSE trades those)."""
    rules = [
        # sunday_to_monday, NOT nearest_workday: when Jan 1 is a Saturday the
        # NYSE still trades Friday Dec 31 (no observance).
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def next_trading_day(after: pd.Timestamp) -> pd.Timestamp:
    return (after + pd.offsets.CustomBusinessDay(1, calendar=NYSEHolidays())).normalize()


def session_close_hour_et(day: pd.Timestamp) -> int:
    """NYSE closes 13:00 ET on Jul 3, the Friday after Thanksgiving, and
    Dec 24 when those are trading days; 16:00 otherwise. (In years where
    those dates are full holidays no bar exists, so returning 13 is harmless.)"""
    d = day.normalize()
    if ((d.month == 7 and d.day == 3) or (d.month == 12 and d.day == 24)
            or (d.month == 11 and d.dayofweek == 4 and 23 <= d.day <= 29)):
        return 13
    return 16
