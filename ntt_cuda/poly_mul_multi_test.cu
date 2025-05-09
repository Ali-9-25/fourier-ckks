#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cuda_runtime.h>
#include "multi_ntt.cu"

struct TestCase {
    uint32_t           mod;
    uint32_t           root;
    uint32_t           root_inv;
    std::vector<uint32_t> A, B, C_expected;
};

static std::string poly_to_string(const std::vector<uint32_t>& a) {
    std::string s;
    int n = (int)a.size();
    for (int i = n-1; i >= 0; --i) {
        uint32_t c = a[i];
        if (!c) continue;
        if (!s.empty()) s += " + ";
        s += std::to_string(c);
        if (i > 0) {
            s += "x";
            if (i > 1) {
                s += "^";
                s += std::to_string(i);
            }
        }
    }
    return s.empty() ? "0" : s;
}

int main() {
    const size_t N = 4;

    std::vector<TestCase> tests = {
        {536871001u, 11u, 146419364u, {51623921u,107100116u,420317839u,529122549u},{1u,0u,0u,1u},{481394806u,223653278u,428066291u,43875469u}},
        {536871017u, 3u, 178957006u, {514852150u,175001745u,12583352u,34364657u},  {1u,0u,0u,1u},{339850405u,162418393u,515089712u,12345790u}},
        {536871089u, 3u, 178957030u, {24888899u,137505621u,368513557u,19284858u},  {1u,0u,0u,1u},{424254367u,305863153u,349228699u,44173757u}},
        {536871233u, 3u, 178957078u, {326213995u,66090711u,416104718u,468974766u}, {1u,0u,0u,1u},{260123284u,186857226u,484001185u,258317528u}},
        {536871337u,10u,375809936u, {28829878u,269046690u,230329201u,48678983u},  {1u,0u,0u,1u},{296654525u,38717489u,181650218u,77508861u}}
    };

    size_t num_polys = tests.size();

    std::vector<uint32_t> A_host, B_host, C_host(N * num_polys);
    std::vector<uint32_t> roots, roots_inv, primes;

    for (auto& test : tests) {
        A_host.insert(A_host.end(), test.A.begin(), test.A.end());
        B_host.insert(B_host.end(), test.B.begin(), test.B.end());
        roots.push_back(test.root);
        roots_inv.push_back(test.root_inv);
        primes.push_back(test.mod);
    }

    poly_mul_multi_gpu(A_host.data(), B_host.data(), C_host.data(), N,
                       roots.data(), roots_inv.data(), primes.data(), num_polys);

    bool all_pass = true;

    for (size_t i = 0; i < num_polys; ++i) {
        auto &t = tests[i];
        std::vector<uint32_t> C_result(C_host.begin() + i*N, C_host.begin() + (i+1)*N);

        bool pass = (C_result == t.C_expected);

        std::cout << "Test " << (i + 1) << ": ";
        if (pass) {
            std::cout << "PASS\n";
        } else {
            all_pass = false;
            std::cout << "FAIL\n"
                      << "  mod=" << t.mod
                      << "  ROOT=" << t.root
                      << "  ROOT_INV=" << t.root_inv << "\n"
                      << "  A(x)= " << poly_to_string(t.A) << "\n"
                      << "  B(x)= " << poly_to_string(t.B) << "\n"
                      << "  Expected: " << poly_to_string(t.C_expected) << "\n"
                      << "  Got     : " << poly_to_string(C_result) << "\n";
        }
    }

    return all_pass ? 0 : 1;
}