import ipdb
import json
import re
from collections import defaultdict

def apply_template(template, data):
    if isinstance(template, str):
        return template.format(**data)
    elif isinstance(template, list):
        prompt = []
        for conversation_turn in template:
            p = conversation_turn.copy()
            p['content'] = p['content'].format(**data)
            prompt.append(p)
        return prompt
    else:
        raise ValueError(f"Unknown template type {type(template)}")

def parse_broken_json(x):
    improved_translation = ""
    errors = defaultdict(list)
    if '"errors": ' in x and "improved translation" in x:
        data = x.split('", "errors": ')
        if len(data) != 2:
            return {"improved translation": improved_translation, "errors": errors}
        # from data[0] parse improved translation
        improved_translation = data[0].split('"improved translation": "')[1]
        # remove last character from data[1]
        data[1] = data[1][:-1]

        try:
            errors = json.loads(data[1])
        except:
            # just try to get error count
            words = re.findall(r'\b\w+\b', data[1].lower())
            keywords = ['critical', 'major', 'minor']

            last_key = None
            for word in words:
                if word in keywords:
                    last_key = word
                elif last_key is not None and word == "class":
                    errors[last_key].append({"class": "other"})

    return {"improved translation": improved_translation, "errors": errors}


def parse_error_class(error):
    # parse error from error description, errors are ['accuracy', 'fluency', 'locale convention', 'style', 'terminology', 'non-translation', 'other']
    #  locale convention (currency, date, name, telephone, or time format), style (awkward), terminology (inappropriate for context, inconsistent use),
    class_name = "unknown"
    if "accuracy" in error:
        class_name = "accuracy"
        for subclass in ["addition", "mistranslation", "omission", "untranslated text"]:
            if subclass in error:
                class_name = f"accuracy-{subclass}"
    elif "fluency" in error:
        class_name = "fluency"
        for subclass in ["character encoding", "grammar", "inconsistency", "punctuation", "register", "spelling"]:
            if subclass in error:
                class_name = f"fluency-{subclass}"
    elif "locale convention" in error:
        class_name = "locale convention"
        for subclass in ["currency", "date", "name", "telephone", "time"]:
            if subclass in error:
                class_name = f"locale convention-{subclass}"
    elif "style" in error:
        class_name = "style"
    elif "terminology" in error:
        class_name = "terminology"
        for subclass in ["inappropriate", "inconsistent"]:
            if subclass in error:
                class_name = f"terminology-{subclass}"
    elif "non-translation" in error:
        class_name = "non-translation"
    elif "other" in error:
        class_name = "other"

    return class_name


def parse_mqm_answer(x, list_mqm_errors=False, full_desc=True):
    if x is None:
        return None

    x = str(x)
    if x.startswith('{"improved translation"'):
        try:
            x = json.loads(x)
        except:
            x = parse_broken_json(x)
        errors = x["errors"]


    else:
        x = x.lower()
        errors = {'critical': [], 'major': [], 'minor': []}
        error_level = None
        # Patched (not upstream): các category MQM hợp lệ theo đúng prompt few-shot.
        # Dùng để lọc "đây có thật là 1 dòng lỗi không" bên dưới -- xem comment ở đó.
        VALID_ERROR_PREFIXES = ('accuracy', 'fluency', 'locale convention', 'style',
                                 'terminology', 'non-translation', 'other')
        for line in x.split('\n'):
            line = line.strip()
            if "no-error" in line or "no error" in line or "" == line:
                continue
            if "critical:" == line:
                error_level = "critical"
                continue
            elif "major:" == line:
                error_level = "major"
                continue
            elif "minor:" == line:
                error_level = "minor"
                continue

            # bỏ gạch đầu dòng/bullet ("- accuracy/omission - ...") trước khi so khớp
            line_content = line.lstrip("-*• ").strip()

            if "critical" in line or "major" in line or "minor" in line:
                if not any(line_content.startswith(p) for p in VALID_ERROR_PREFIXES):
                    print(line)

            if error_level is None:
                print(f"No error level for {line}")
                continue

            # Patched (not upstream): CHỈ tính là lỗi thật nếu dòng thực sự bắt đầu
            # bằng 1 category MQM hợp lệ. Bug gốc: các câu KHẲNG ĐỊNH KHÔNG CÓ LỖI mà
            # model hiện đại hay dùng ("no other major errors", "no critical errors
            # identified", "no minor errors detected beyond the above") không khớp
            # đúng chuỗi "no-error"/"no error" ở check phía trên, nên lọt xuống đây và
            # bị append thẳng vào errors[error_level] -- tức 1 câu Ý NGHĨA TÍCH CỰC lại
            # bị tính thành 1 lỗi CRITICAL/MAJOR/MINOR thật, kéo điểm sập sàn -25 dù
            # bản dịch tốt. Yêu cầu prefix hợp lệ mới cho qua sẽ chặn đứng các câu này.
            if not any(line_content.startswith(p) for p in VALID_ERROR_PREFIXES):
                continue

            if "non-translation" in line:
                errors["critical"].append(line)
            else:
                errors[error_level].append(line)

    error_classes = defaultdict(list)
    final_score = 0
    error_counter = 0
    for error_level in ['critical', 'major', 'minor']:
        if error_level not in errors:
                continue
        for error in errors[error_level]:
            if error_counter < 5 and not list_mqm_errors:
                final_score += 25 if error_level == 'critical' else 5 if error_level == 'major' else 1
                error_counter += 1

            if full_desc:
                error_classes[error_level].append(error)
            else:
                class_name = parse_error_class(error)
                error_classes[error_level].append(class_name)
    if final_score > 25:
        final_score = 25

    if list_mqm_errors:
        return error_classes
    else:
        # negative score is to normalize that higher score is better
        return -final_score


def mqm_fewshot(few_shots):
    prompts = [
        {
            "role": "system",
            "content": f"You are an annotator for the quality of machine translation. Your task is to identify errors and assess the quality of the translation."
        }
    ]

    template = """{source_lang} source:
```{source_seg}```
{target_lang} translation:
```{target_seg}```

Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension."""
   
    for shot in few_shots:
        prompts.append({
            "role": "user",
            "content": template.format(**shot)
        })
        answer = shot['answer']

        prompts.append({
            "role": "assistant",
            "content": answer
        })

    prompts.append({
            "role": "user",
            "content": template
        })

    return prompts


few_shots = {
    "ende": {
            "source_lang": "English",
            "source_seg": "I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.",
            "target_lang": "German",
            "target_seg": "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.",
            "answer": """Critical:
no-error
Major:
accuracy/mistranslation - "involvement"
accuracy/omission - "the account holder"
Minor:
fluency/grammar - "wäre"
fluency/register - "dir"
""",
        },
    "encs": {
            "source_lang": "English",
            "source_seg": "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.",
            "target_lang": "Czech",
            "target_seg": "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.",
            "answer": """Critical:
no-error
Major:
accuracy/addition - "ve Vídni"
accuracy/omission - "the stop-start"
Minor:
terminology/inappropriate for context - "partaje"
""",
        },
    "zhen": {
            "source_lang": "Chinese",
            "source_seg": "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",
            "target_lang": "English",
            "target_seg": "Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.",
            "answer": """Critical:
accuracy/addition - "of high-speed rail"
Major:
accuracy/mistranslation - "go to the reviews"
Minor:
style/awkward - "etc.,"
""",
        },
}

TEMPLATE_GEMBA_MQM = mqm_fewshot([few_shots['ende'], few_shots['encs'], few_shots['zhen']])

