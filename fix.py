content = open('metrics.py', 'r', encoding='utf-8').read()
content = content.replace('pen += float(f.weight) * float(lateness ** 2) * 5.0\n\n\ndef', 'pen += float(f.weight) * float(lateness ** 2) * 5.0\n\n    return {"deadline_penalty": float(pen), "deadline_miss": float(miss), "deadline_total": float(total)}\n\n\ndef')
open('metrics.py', 'w', encoding='utf-8').write(content)
