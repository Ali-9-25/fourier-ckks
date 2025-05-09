/**********************************************************************
  Runs 5 hard-coded tests (from lolo) and prints PASS/FAIL for each.
 *********************************************************************/

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cuda_runtime.h>
#include "ntt.cu"

struct TestCase {
    uint32_t           mod;
    uint32_t           root;
    uint32_t           root_inv;
    std::vector<uint32_t> A, B, C_expected;
};

// helper to pretty-print a length-4 polynomial
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
        // 1)
        {536871001u, 11u, 146419364u,
         /* A */ {51623921u, 107100116u, 420317839u, 529122549u},
         /* B */ {1u, 0u, 0u, 1u},
         /* C */ {481394806u, 223653278u, 428066291u,   43875469u}
        },
        // 2)
        {536871017u, 3u, 178957006u,
         {514852150u, 175001745u, 12583352u, 34364657u},
         {1u,0u,0u,1u},
         {339850405u,162418393u,515089712u,   12345790u}
        },
        // 3)
        {536871089u,3u,178957030u,
         {24888899u,137505621u,368513557u,19284858u},
         {1u,0u,0u,1u},
         {424254367u,305863153u,349228699u,   44173757u}
        },
        // 4)
        {536871233u,3u,178957078u,
         {326213995u,66090711u,416104718u,468974766u},
         {1u,0u,0u,1u},
         {260123284u,186857226u,484001185u,  258317528u}
        },
        // 5)
        {536871337u,10u,375809936u,
         {28829878u,269046690u,230329201u,48678983u},
         {1u,0u,0u,1u},
         {296654525u, 38717489u,181650218u,  77508861u}
        }
    };

    bool all_pass = true;
    for (size_t i = 0; i < tests.size(); ++i) {
        auto &t = tests[i];
        std::vector<uint32_t> C(N);
        int err = poly_mul_gpu(
            t.A.data(), t.B.data(), C.data(),
            N, t.root, t.root_inv, t.mod);
        cudaDeviceSynchronize();

        bool pass = (err == 0);
        if (pass) {
            for (size_t j = 0; j < N; ++j) {
                if (C[j] != t.C_expected[j]) {
                    pass = false;
                    break;
                }
            }
        }

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
                      << "  Got     : " << poly_to_string(C) << "\n";
        }
    }

    return all_pass ? 0 : 1;
}
