load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "network-example-app",
    srcs = ["network-example-app.cpp"],
    deps = [
        "//common:data-utils",
        "//network:neural-network",
        "@eigen",
    ],
)

cc_binary(
    name = "pool-example-app",
    srcs = ["pool-example-app.cpp"],
    deps = [
        "//common:data-utils",
        "//pool:neural-pool",
        "@eigen",
    ],
)
