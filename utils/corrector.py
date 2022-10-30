import re
from pyaspeller import YandexSpeller
from fuzzywuzzy import fuzz
import jamspell

def postprocessing(text, threshold, model_ru='yandex', model_en='yandex'):
    text = re.sub('[@#$₽]', '', text)

    # lang detector

    if bool(re.search('[а-яА-Я]', text)):
        lang = 'ru'
    else:
        lang = 'en'

    # spell checker

    if lang == 'en':
        if model_en == 'yandex':
            speller = YandexSpeller()
            fixed = speller.spelled(text)
        if model_en == 'jamspell':
            corrector = jamspell.TSpellCorrector()
            corrector.LoadLangModel("en.bin")
            fixed = corrector.FixFragment(text)

    if lang == 'ru':
        if model_ru == 'yandex':
            speller = YandexSpeller()
            fixed = speller.spelled(text)
        if model_ru == 'jamspell':
            corrector = jamspell.TSpellCorrector()
            corrector.LoadLangModel("ru_small.bin")
            fixed = corrector.FixFragment(text)

    # check similarity

    if fuzz.ratio(text, fixed) > threshold:
        return fixed
    else:
        return text


def func(x):
    preds_new = []
    for p in range(len(x)):
        cur_string = x[p]
        cur_string = re.sub('[@#$₽]', '', cur_string)
        new_string = postprocessing(cur_string, 70, model_ru='yandex', model_en='yandex')
        preds_new.append(new_string)
    return preds_new
