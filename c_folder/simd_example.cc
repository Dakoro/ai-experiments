#include <iostream>
#include <vector>
#include <immintrin.h> // Pour AVX

// Addition de deux vecteurs en utilisant les intrinsèques AVX
// Suppose que la taille du vecteur est un multiple de 8 (pour __m256 qui contient 8 floats)
void simd_add_vectors(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& result) {
    size_t size = a.size();
    for (size_t i = 0; i < size; i += 8) {
        // Charger 8 floats de chaque vecteur dans des registres AVX
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_b = _mm256_loadu_ps(&b[i]);
        
        // Additionner les deux registres
        __m256 vec_res = _mm256_add_ps(vec_a, vec_b);
        
        // Stocker le résultat
        _mm256_storeu_ps(&result[i], vec_res);
    }
}

int main() {
    std::vector<float> a(256, 1.0f);
    std::vector<float> b(256, 2.0f);
    std::vector<float> result(256);

    simd_add_vectors(a, b, result);
    std::cout << "Exemple d'addition SIMD: result[0] = " << result[0] << std::endl; // Devrait être 3.0
    return 0;
}