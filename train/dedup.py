import os
from ovos_utils.bracket_expansion import expand_template
from ovos_utils.list_utils import flatten_list, deduplicate_list


for f in os.listdir('.'):
    if not f.endswith('.txt'):
        continue
    with open(f) as fi:
        lines = fi.read().split("\n")
        if not f.startswith('keywords'):
            lines = flatten_list(expand_template(l.lower().strip(".?"))
                                 for l in lines if "{keyword}" in l)
        lines = deduplicate_list(lines)
        print(lines)

    with open(f, "w") as fi:
        fi.write("\n".join(sorted(lines)))