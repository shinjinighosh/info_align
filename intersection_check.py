with open("tasks/lex_trans/lex/es/train.txt") as train_file:
    res = train_file.readlines()
    wordset_tr = {line.strip().split()[0] for line in res}
    print(sorted(list(wordset_tr))[:100])
    train_file.close()

with open("tasks/lex_trans/lex/es/test.txt") as test_file:
    res = test_file.readlines()
    wordset_test = {line.strip().split()[0] for line in res}
    print(sorted(list(wordset_test))[:100])
    test_file.close()

print(wordset_tr.intersection(wordset_test))
