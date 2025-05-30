// helper macro for template specializations
#define SPECIALIZE_CASE(T, index_t, signWrite, signRead, SH, U, FU, D, FD, MODE, TW, TH, W, XR, WS) \
    template void run_filtered_lrelu_kernel<T, index_t, signWrite, signRead, SH, MODE, U, FU, D, FD, TW, TH, W, XR, WS>(filtered_lrelu_kernel_params& p);

// list of all the kernel cases
CASE(/*sharedKB*/48, /*up,fu*/1,1,  /*down,fd*/1,1,  /*mode*/MODE_FUFD, /*tw,th,warps,xrep,wskip*/64,  178, 32,  0,  0) // 1t-upf1-downf1
CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/1,1,  /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/152, 95,  16,  0,  0) // 4t-ups2-downf1
CASE(/*sharedKB*/48, /*up,fu*/1,1,  /*down,fd*/2,8,  /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/56,  22,  16,  0,  0) // 4t-upf1-downs2
CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/2,8,  /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/56,  29,  16,  11, 0) // 4t-ups2-downs2
CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/2,8,  /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/60,  28,  16,  0,  0) // 4t-upf2-downs2
CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/2,8,  /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/56,  28,  16,  0,  0) // 4t-ups2-downf2
CASE(/*sharedKB*/48, /*up,fu*/4,16, /*down,fd*/2,8,  /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/56,  31,  16,  11, 0) // 4t-ups4-downs2
CASE(/*sharedKB*/48, /*up,fu*/4,16, /*down,fd*/2,8,  /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/56,  36,  16,  0,  0) // 4t-ups4-downf2
CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/4,16, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/16,  22,  16,  12, 0) // 4t-ups2-downs4
CASE(/*sharedKB*/48, /*up,fu*/2,8,  /*down,fd*/4,16, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/29,  15,  16,  0,  0) // 4t-upf2-downs4
CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/1,1,  /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/96,  150, 28,  0,  0) // 6t-ups2-downf1
CASE(/*sharedKB*/48, /*up,fu*/1,1,  /*down,fd*/2,12, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/32,  35,  24,  0,  0) // 6t-upf1-downs2
CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/2,12, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/32,  46,  16,  10, 0) // 6t-ups2-downs2
CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/2,12, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/58,  28,  24,  8,  0) // 6t-upf2-downs2
CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/2,12, /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/52,  28,  16,  0,  0) // 6t-ups2-downf2
CASE(/*sharedKB*/48, /*up,fu*/4,24, /*down,fd*/2,12, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/32,  51,  16,  5,  0) // 6t-ups4-downs2
CASE(/*sharedKB*/48, /*up,fu*/4,24, /*down,fd*/2,12, /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/32,  56,  16,  6,  0) // 6t-ups4-downf2
CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/4,24, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/16,  18,  16,  12, 0) // 6t-ups2-downs4
CASE(/*sharedKB*/96, /*up,fu*/2,12, /*down,fd*/4,24, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/27,  31,  32,  6,  0) // 6t-upf2-downs4 96kB
CASE(/*sharedKB*/48, /*up,fu*/2,12, /*down,fd*/4,24, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/27,  13,  24,  0,  0) // 6t-upf2-downs4
CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/1,1,  /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/148, 89,  24,  0,  0) // 8t-ups2-downf1
CASE(/*sharedKB*/48, /*up,fu*/1,1,  /*down,fd*/2,16, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/32,  31,  16,  5,  0) // 8t-upf1-downs2
CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/2,16, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/32,  41,  16,  9,  0) // 8t-ups2-downs2
CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/2,16, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/56,  26,  24,  0,  0) // 8t-upf2-downs2
CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/2,16, /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/32,  40,  16,  0,  0) // 8t-ups2-downf2
CASE(/*sharedKB*/48, /*up,fu*/4,32, /*down,fd*/2,16, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/32,  46,  24,  5,  0) // 8t-ups4-downs2
CASE(/*sharedKB*/48, /*up,fu*/4,32, /*down,fd*/2,16, /*mode*/MODE_SUFD, /*tw,th,warps,xrep,wskip*/32,  50,  16,  0,  0) // 8t-ups4-downf2
CASE(/*sharedKB*/96, /*up,fu*/2,16, /*down,fd*/4,32, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/24,  24,  32,  12, 1) // 8t-ups2-downs4 96kB
CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/4,32, /*mode*/MODE_SUSD, /*tw,th,warps,xrep,wskip*/16,  13,  16,  10, 1) // 8t-ups2-downs4
CASE(/*sharedKB*/96, /*up,fu*/2,16, /*down,fd*/4,32, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/25,  28,  28,  4,  0) // 8t-upf2-downs4 96kB
CASE(/*sharedKB*/48, /*up,fu*/2,16, /*down,fd*/4,32, /*mode*/MODE_FUSD, /*tw,th,warps,xrep,wskip*/25,  10,  24,  0,  0) // 8t-upf2-downs4
