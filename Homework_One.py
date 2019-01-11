# !/usr/bin/env python
# -*- coding: utf-8 -*-
print("")
print(" Добро пожаловать в программу для домашней работ")
print(" Здесь показаны три различных вида ДЗ")
print(" Выбирайте, на что будите смотреть:")
print("")
print(" 1 - Poker")
print(" 2 - Deco")
print(" 3 - Log_Analyzer")
print(" 4 - Выход из программы")
print("")
while True:
    a = input("Повторите действие:")
    try:
        a = int(a)
        if int(a) >= 1 and int(a)<=4:
            break
    except ValueError:
        pass
if a == 4:
    raise exit(1)
if a == 1:
    print("")
    print("   _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")
    print(" # Реализуйте функцию Best_hand, которая принимает на вход ")
    print(" # покерную 'руку' (hand) из 7ми карт и возвращает лучшую")
    print(" # (Относительно значения, возвращаемого hand_rank)")
    print(" # 'руку' из 5ти карт. У каждой карты есть масть(suit) и")
    print(" # ранг(rank)")
    print("")
    print(" # Масти: Трефы(Clubs, C), пики(Spades, S), черви(Hearts, H), бубны(Diamonds, D)")
    print(" # Ранги: 2,3,4,5,6,7,8,9,10(Ten, T), Валет(Jack, J), Дама(queen, Q), Король(King, K), Туз(Ace, A)")
    print(" # Например: AS - Туз пик(Ace od Spades), TH - Десятка червь(Ten of Hearts), 3C - Тройка треф(Three of Clubs)")
    print("")
    print(" # Задание со *")
    print(" # Реализуйте функцию Best_Wild_Hand, которая принимает на вход")
    print(" # покерную 'руку'(hand) из 7ми карт и возвращает лучшую")
    print(" # (Относительно значения, возвращаемого hand_rank)")
    print(" # 'руку' из 5ти карт. Кроме прочего в данном варианте 'рука'")
    print(" # может включать джокера. Джокеры могут заменить карту любой")
    print(" # масти и ранга того же цвета. Чёрный джокер '?B' может быть")
    print(" # использован в качестве треф или пик любого ранга, красный ")
    print(" # джокер '?R' - в качестве черв и бубен любого ранга.")
    print("")
    print(" # Одна функция уже реализована, сигнатуры и описания других даны.")
    print(" # Вам неверняка пригодится itertools.")
    print(" # Можно свободно определять свои функции и т.п.")
    print("   _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")
    print("")
    import itertools
    from collections import Counter

    def hand_rank(hand):
        """Возвращает значение определяющее ранг 'руки'"""
        ranks = card_ranks(hand)
        if straight(ranks) and flush(hand):
            return (8, max(ranks))
        elif kind (4, ranks):
            return (7, kind(4, ranks), kind(1, ranks))
        elif kind (3, ranks) and kind (2, ranks):
            return (6, (3, ranks) and kind (2, ranks))
        elif flush(hand):
            return (5, ranks)
        elif straight(ranks):
            return (4, max(ranks))
        elif kind(3, ranks):
            return (3, kind(3, ranks), ranks)
        elif two_pair(ranks):
            return (2, two_pair(ranks), ranks)
        elif kind(2, ranks):
            return (1, kind(2, ranks), ranks)
        else:
            return (0, ranks)

    def card_ranks(hand):
        """Возвращает список рангов, отчортированный от большого к меньшему"""
        return sorted(["23456789TJQKA".index(rank) for rank,suit in hand], reverse = True)

    def flush(hand):
        """Возвращает TRUE, если все карты одной масти"""
        return len(set([suit for rank, suit in hand])) == 1

    def straight(ranks):
        """Возвращает TRUE, если отсортированные ранги формируют последовательность 5ти,
        где у 5ти карт ранги идут по порядку (стрит)"""
        return ranks == list(range(max(ranks), min(ranks)-1, -1))

    def kind(n, ranks):
        """Возвращает первый ранг, который n раз встречается в данной руке
        Возвращает None, если ничего не найдено"""
        for rank, group in itertools.groupby (ranks):
            if len(list(group)) == n:
                return rank
        return None

    def two_pair(ranks):
        """Если есть две пары, то возвращает два соответствующих ранга
        Иначе возвращает None"""
        r1, r2 = kind(2, ranks), kind(2, ranks[::-1])
        return (r1, r2) if r1 and (r1 != r2) else None

    def best_hand(hand):
        """Из 'руки' в 7 карт возвращает лучшую 'руку' в 5 карт"""
        handa5 = itertools.combinations(hand, 5)
        return max(handa5, key = hand_rank)

    def best_wild_hand(hand):
        """Best_Hand но с джокерами"""
        jokers = {"?B" : "CS", "?R" : "HD"}
        clear_hand = [card for card in hand if card not in jokers]
        joker_suits = [jokers[card] for card in hand if card in jokers]
        hands = [clear_hand]
        for joker_suit in joker_suits:
            cards = ["%s%s"%(r, s) for r,s in itertools.product("23456789TJQKA", joker_suit)]
            cards = [card for card in cards if card not in clear_hand]
            hands = [h + [card] for h in hands for card in cards]
        return max(set([best_hand(h) for h in hands]), key=hand_rank)

    def test_best_hand():
        print("test_best_hand...")
        assert (sorted(best_hand("6C 7C 8C 9C TC 5C JS".split()))
                == ['6C', '7C', '8C', '9C', 'TC'])
        assert (sorted(best_hand("TD TC TH 7C 7D 8C 8S".split()))
                == ['8C', '8S', 'TC', 'TD', 'TH'])
        assert (sorted(best_hand("JD TC TH 7C 7D 7S 7H".split()))
                == ['7C', '7D', '7H', '7S', 'JD'])
        print("OK")

    def test_best_wild_hand():
        print("test_best_wild_hunt...")
        assert (sorted(best_wild_hand("6C 7C 8C 9C TC 5C ?B".split()))
                == ['7C', '8C', '9C', 'JC', 'TC'])
        assert (sorted(best_wild_hand("TD TC 5H 5C 7C ?R ?B".split()))
                == ['7C', 'TC', 'TD', 'TH', 'TS'])
        assert (sorted(best_wild_hand("JD TC TH 7C 7D 7S 7H".split()))
                == ['7C', '7D', '7H', '7S', 'JD'])
        print("OK")

    if __name__ == '__main__':
        test_best_hand()
        test_best_wild_hand()

    input("")

if a == 2:
    print("")
    from functools import update_wrapper
    from functools import wraps


    def disable(f):
        '''
        Disable a decorator by re-assigning the decorator's name
        to this function. For example, to turn off memoization:
        >>> memo = disable
        '''
        return f if callable(f) else lambda f: f


    def decorator(deco):
        '''
        Decorate a decorator so that it inherits the docstrings
        and stuff from the function it's decorating.
        '''
        def wrapper(f):
            return update_wrapper(deco(f), f)
        return update_wrapper(wrapper, deco)


    def countcalls(f):
        '''Decorator that counts calls made to the function decorated.'''
        @wraps(f)
        def wrapper(*args, **kwargs):
            wrapper.calls += 1
            return f(*args, **kwargs)
        wrapper.calls = 0
        return wrapper


    def memo(f):
        '''
        Memoize a function so that it caches all return values for
        faster future lookups.
        '''
        cache = {}
        @wraps(f)
        def wrapper(*args):
            if args not in cache:
                cache[args] = f(*args)
            update_wrapper(wrapper, f)
            return cache[args]
        return wrapper


    def n_ary(f):
        '''
        Given binary function f(x, y), return an n_ary function such
        that f(x, y, z) = f(x, f(y,z)), etc. Also allow f(x) = x.
        '''
        @wraps(f)
        def wrapper(x, *args):
            return x if not args else f(x, wrapper(*args))
        return wrapper


    def trace(fill_value):
        '''Trace calls made to function decorated.
        @trace("____")
        def fib(n):
            ....
        >>> fib(3)
         --> fib(3)
        ____ --> fib(2)
        ________ --> fib(1)
        ________ <-- fib(1) == 1
        ________ --> fib(0)
        ________ <-- fib(0) == 1
        ____ <-- fib(2) == 2
        ____ --> fib(1)
        ____ <-- fib(1) == 1
         <-- fib(3) == 3
        '''
        def trace_decorator(f):
            @wraps(f)
            def wrapper(*args):
                prefix = fill_value * wrapper.level
                fargs = ", ".join(str(a) for a in args)
                print ("{} --> {}({})".format(prefix, f.__name__, fargs))
                wrapper.level += 1
                result = f(*args)
                print ("{} <-- {}({}) == {}".format(prefix, f.__name__, fargs, result))
                wrapper.level -= 1
                return result
            wrapper.level = 0
            return wrapper
        return trace_decorator


    @memo
    @countcalls
    @n_ary
    def foo(a, b):
        return a + b


    @countcalls
    @memo
    @n_ary
    def bar(a, b):
        return a * b


    @countcalls
    @trace("####")
    @memo
    def fib(n):
        return 1 if n <= 1 else fib(n-1) + fib(n-2)


    def main():
        print (foo(4, 3))
        print (foo(4, 3, 2))
        print (foo(4, 3))
        print ("foo was called", foo.calls, "times")

        print (bar(4, 3))
        print (bar(4, 3, 2))
        print (bar(4, 3, 2, 1))
        print ("bar was called", bar.calls, "times")

        print (fib.__doc__)
        fib(3)
        print (fib.calls, 'calls made')


    if __name__ == '__main__':
        main()
    input("")
if a == 3:
    print("")
    import argparse
    import pathlib
    from pathlib import Path
    import json
    import logging
    import datetime
    import re
    import gzip
    import collections
    import statistics
    import string
    import os
    from typing import NamedTuple, Union, Optional, List, Dict, Any, cast

    default_config = {
        "REPORT_SIZE": 1000,
        "REPORT_DIR": "./reports",
        "LOG_DIR": "D:\Задания питона\log",
        "LOG_FILE": None,
        "ERRORS_TRESHOLD": 0.01,
        "TS_FILE": "D:\Задания питона\log\nginx-access-ui.log",
    }

    log_pattern = re.compile(
        r"(?P<remote_addr>[\d\.]+)\s"
        r"(?P<remote_user>\S*)\s+"
        r"(?P<http_x_real_ip>\S*)\s"
        r"\[(?P<time_local>.*?)\]\s"
        r'"(?P<request>.*?)"\s'
        r"(?P<status>\d+)\s"
        r"(?P<body_bytes_sent>\S*)\s"
        r'"(?P<http_referer>.*?)"\s'
        r'"(?P<http_user_agent>.*?)"\s'
        r'"(?P<http_x_forwarded_for>.*?)"\s'
        r'"(?P<http_X_REQUEST_ID>.*?)"\s'
        r'"(?P<http_X_RB_USER>.*?)"\s'
        r"(?P<request_time>\d+\.\d+)\s*"
    )

    Config  = Dict[str, Any]
    Log = NamedTuple('Log', [('path', pathlib.Path), ('date', datetime.date), ('ext', str)])
    Request = NamedTuple('Request', [('url', str), ('request_time', float)])

    def update_ts(ts_file: pathlib.Path) -> None:
        now = datetime.datetime.now()
        timestamp = int(now.timestamp())
        ts_file.write_text(str(timestamp))
        os.utime(ts_file.absolute(), times=(timestamp, timestamp))

    def create_report(template_path: pathlib.Path, destination_path: pathlib.Path, log_statistics: List[Dict[str, Union[str, float]]]) -> None:
        with template_path.open() as f:
            template = string.Template(f.read())
        report = template.safe_substitute(table_json=json.dumps(log_statistics))
        with destination_path.open(mode='w') as f:
            f.write(report)

    def process_line(line: str) -> Optional[Request]:
        m = log_pattern.match(line)
        if not m:
            return None
    
        log_line = m.groupdict()
        try:
            method, url, protocol = log_line['request'].split()
            request_time = float(log_line['request_time'])
        except (ValueError, TypeError):
            return None
        else:
            return Request(url, request_time)

    def process_log(log: Log, errors_treshold: float) -> List[Dict[str, Union[str, float]]]:
        if log.ext == '.gz':
            f = gzip.open(log.path.absolute(), mode='rt')
        else:
            f = log.path.open()
    
        n_loglines = 0
        n_fails = 0
        url2times: Dict[str, List[float]] = collections.defaultdict(list)
        with f:
            for line in f:
                n_loglines += 1
                request = process_line(line)
                if not request:
                    n_fails += 1
                    continue
                url2times[request.url].append(request.request_time)

        errors = n_fails / n_loglines
        if errors > errors_treshold:
            raise Exception(f"Доля ошибок {errors} превышает {errors_treshold}")

        total_count = 0
        total_time = 0.
        for request_times in url2times.values():
            total_count += len(request_times)
            total_time  += sum(request_times)
    
        stat = []
        for url, request_times in url2times.items():
            stat.append({
                'url': url,
                'count': len(request_times),
                'count_perc': round(100. * len(request_times) / float(total_count), 3),
                'time_sum': round(sum(request_times), 3),
                'time_perc': round(100. * sum(request_times) / total_time, 3),
                'time_avg': round(statistics.mean(request_times), 3),
                'time_max': round(max(request_times), 3),
                "time_med": round(statistics.median(request_times), 3),
            })
    
        return stat # type: ignore

    def get_report_path(report_dir: pathlib.Path, log: Log) -> pathlib.Path:
        if not report_dir.exists() or not report_dir.is_dir():
            raise FileNotFoundError("Неверно указан путь к директории с отчетами")
    
        report_filename = f'report-{log.date:%Y.%m.%d}.html'
        report_path = report_dir / report_filename
        return report_path

    def get_last_logfile(log_dir: pathlib.Path) -> Optional[Log]:
        p = Path ('D:\Задания питона\nginx-access-ui.log-20170630')
        if not log_dir.exists() or not log_dir.is_dir():
            raise FileNotFoundError("Неверно указан путь к директории с журналами")
    
        logfile = None
        pattern = re.compile(r"^nginx-access-ui\.log-(\d{8})(\.gz)?$")
        for path in log_dir.iterdir():
            try:
                [(date, ext)] = re.findall(pattern, str(path))
                log_date = datetime.datetime.strptime(date, "%Y%m%d").date()
                if not logfile or logfile.date > log_date:
                    logfile = Log(path, log_date, ext)
            except ValueError:
                pass
    
        return logfile

    def setup_logging(logfile: Optional[str]) -> None:
        logging.basicConfig( # type: ignore
            level=logging.INFO,
            format="[%(asctime)s] %(levelname).1s %(message)s",
            datefmt="%Y.%m.%d %H:%M:%S",
            filename=logfile)

    def get_config(path: str, default_config: Config) -> Config:
        if not path:
            return default_config
    
        p = pathlib.Path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError("Неверно указан путь к конфигурационному файлу")
    
        with p.open() as f:
            config = json.load(f)
    
        return {**default_config, **config}

    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser("Обработка лог-файлов и генерирование отчета")
        parser.add_argument("--config",
            dest="config_path",
            help="Путь к конфигурационному файлу")
        return parser.parse_args()

    def main(config: Config) -> None:
        log_dir = pathlib.Path(cast(str, config.get("LOG_DIR")))
        last_log = get_last_logfile(log_dir)
        if not last_log:
            logging.info(f"Нет логов в '{log_dir}' для обработки")
            return

        report_dir = pathlib.Path(cast(str, config.get("REPORT_DIR")))
        report_path = get_report_path(report_dir, last_log)
        if report_path.exists():
           logging.info(f"Отчет для '{last_log.path}' уже существует")
           return
        
        log_statistics = process_log(last_log, cast(float, config.get("ERRORS_TRESHOLD")))
        log_statistics = sorted(log_statistics, key=lambda r: r['time_sum'], reverse=True)
        log_statistics = log_statistics[:config.get("REPORT_SIZE")]
        report_template_path = report_dir / "report.html"
        create_report(report_template_path, report_path, log_statistics)
    
        ts_path = pathlib.Path(cast(str, config.get('TS_FILE')))
        update_ts(ts_path)

    if __name__ == "__main__":
        args = parse_args()
        config = get_config(args.config_path, default_config)
        setup_logging(config.get('LOG_FILE'))

        try:
            main(config)
        except Exception as e:
            logging.exception(str(e))
    input("")
