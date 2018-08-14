chars_to_escape = [ '_' ]

def escape_chars(s : str):
    for e in chars_to_escape:
        s = s.replace(e, '\\'+e)

    return s


def trimmed_float(f, float_length = 5):
    if type(f) is list:
        return list([str(round(ff, float_length)) for ff in f])
    else:
        return str(round(f, float_length))


def dict_zip(*dicts):
    return {k: [d[k] for d in dicts] for k in dicts[0].keys()}


def dict_to_tex_table(d : dict, float_length = 5):
    if len(d) == 0:
        return

    first_elem = next(iter(d.values()))
    if type(first_elem) is list:
        table_length = len(first_elem) + 1
    else:
        table_length = 2

    header = r'\begin{tabular}{' + ('|c' * table_length) + '|}\n' + \
        '\t \\hline' + '\n'
    
    body = ""
    for k,v in d.items():
        if type(v) is list:
            body = body + '\t\t' + escape_chars(str(k)) + \
                   ' & ' + escape_chars(" & ".join(trimmed_float(v, float_length))) + ' \\\\ \n'
        else:
            body = body + '\t\t' + escape_chars(str(k)) + \
                   ' & ' + escape_chars(trimmed_float(v, float_length)) + ' \\\\ \n'

    footer = '\t\\hline' + '\n' + \
        r'\end{tabular}' + '\n'

    return header + body + footer
