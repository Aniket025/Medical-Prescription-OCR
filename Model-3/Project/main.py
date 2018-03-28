import preprocess
import GCVocr
import spellcorrection

import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filename = argv[1]

    PP = Preprocess(filename)
    image = PP.read_image()
    crop_image = PP.crop_image(image)
    gray_image = PP.binarization(crop_image)

    plt.show(image)
    plt.show(crop_image)
    plt.show(gray_image)

    cv2.imwrite("temp.jpg", gray_image)

    OCR = GoogleCloudVisionOCR("temp.jpg")
    response = OCR.request_ocr()
    if response.status_code != 200 or response.json().get('error'):
        print(response.text)
        return
    else:
        for idx, resp in enumerate(response.json()['responses']):
            imgname = image_filenames[idx]
            jpath = join(OCR.RESULTS_DIR, basename(imgname) + '.json')
            with open(jpath, 'w') as f:
                datatxt = json.dumps(resp, indent=2)
                print("Wrote", len(datatxt), "bytes to", jpath)
                f.write(datatxt)

            print("---------------------------------------------")
            t = resp['textAnnotations'][0]
            print("    Bounding Polygon:")
            print(t['boundingPoly'])
            print("    Text:")
            print(t['description'])

            ss = SymSpell(max_edit_distance=2)
            with open('./bad-words.csv') as bf:
                bad_words = bf.readlines()
            bad_words = [word.strip() for word in bad_words]

            # fetch english words dictionary
            with open('./english_words_479k.txt') as f:
                words = f.readlines()
            eng_words = [word.strip() for word in words]
            print(eng_words[:5])
            print(bad_words[:5])

            print('Total english words: {}'.format(len(eng_words)))
            print('Total bad words: {}'.format(len(bad_words)))

            print('create symspell dict...')

            if to_sample:
                # sampling from list for kernel runtime
                sample_idxs = random.sample(range(len(eng_words)), 100)
                eng_words = [eng_words[i] for i in sorted(sample_idxs)] + \
                    'to infinity and beyond'.split() # make sure our sample misspell is in there

            all_words_list = list(set(bad_words + eng_words))
            silence = ss.create_dictionary_from_arr(all_words_list, token_pattern=r'.+')

            # create a dictionary of rightly spelled words for lookup
            words_dict = {k: 0 for k in all_words_list}

            sample_text = 'to infifity and byond'
            tokens = spacy_tokenize(sample_text)

            print('run spell checker...')
            print()
            print('original text: ' + sample_text)
            print()
            correct_text = spell_corrector(tokens, words_dict)
            print('corrected text: ' + correct_text)

            print('Done.')
            print("--- %s seconds ---" % (time.time() - start_time))
