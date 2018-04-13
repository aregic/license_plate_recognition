import sys

def progress_bar(count, total, prefix='', suffix=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    if count != total:
        sys.stdout.write('%s [%s] %s%s ...%s\r' % (prefix, bar, percents, '%', suffix))
    else:
        print('%s [%s] %s%s ...%s\r' % (prefix, bar, percents, '%', suffix))
    sys.stdout.flush()
