load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "neural-pool",
    srcs = ["neural_pool.cpp"],
    hdrs = ["neural_pool.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common:data-utils",
        "@eigen",
    ],
)
