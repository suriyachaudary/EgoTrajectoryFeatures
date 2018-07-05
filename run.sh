cd make_vocab
cmake .
make
./make_vocab ../train_windows.txt bidt_30_ temp
cd ../biDT_temporal
cmake .
make
./biDT_temporal ../train_windows.txt ../test_windows.txt ../make_vocab bidt_30_ train_features test_features temp vocab_sizes