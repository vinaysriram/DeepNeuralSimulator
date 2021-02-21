load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "neural-network",
    srcs = ["neural_network.cpp"],
    hdrs = ["neural_network.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//common:data-utils",
        "@eigen",
    ],
)
