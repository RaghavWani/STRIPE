#g++ -std=c++17 ./rfi_mitigation.cpp -o stripe
#g++ -O3 -march=native -std=c++17 ./rfi_mitigation.cpp -o stripe
g++ -O3 -march=native -ffast-math -fopenmp -std=c++17 ./rfi_mitigation.cpp -o stripe
