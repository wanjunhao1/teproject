content = open('metrics.py', 'r', encoding='utf-8').read()
content = content.replace('pen += float(f.weight) * float(lateness ** 2) * 0.05', 'pen += 0.1 * float(f.weight) * float(lateness)')
open('metrics.py', 'w', encoding='utf-8').write(content)
