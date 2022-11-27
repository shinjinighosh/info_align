import numpy as np
from tasks import lex_trans


def main():
    random = np.random.RandomState(0)

    data, vocab = lex_trans.load_all()
    test_data, test_vocab = lex_trans.load_test()
    print(len(data), len(test_data))

    translation_dict = {}

    for en, es in test_data:
        en = test_vocab.decode(en)
        es = test_vocab.decode(es)
        if en in translation_dict:
            translation_dict[en].append(es)
        else:
            translation_dict[en] = [es]

    print(translation_dict)


if __name__ == "__main__":
    main()
