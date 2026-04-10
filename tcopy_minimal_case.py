import tilelang
import tilelang.language as T
import torch


tilelang.cache.clear_cache()
torch.set_default_device("npu")


@tilelang.jit(out_idx=[1, 2])
def ub_copy_repro(N: int = 513, dtype: str = "float32"):
    @T.prim_func
    def main(
        A: T.Tensor((1, N), dtype),
        B_relay: T.Tensor((1, N), dtype),
        B_direct: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            in_ub = T.alloc_ub((1, N), dtype)
            relay_ub = T.alloc_ub((1, N), dtype)

            with T.Scope("V"):
                T.copy(A[0, :], in_ub)

                T.printf("debug in_ub\n")
                T.dump_tensor(in_ub, 2001, N, (1, N))

                T.copy(in_ub, relay_ub)

                T.printf("debug relay_ub\n")
                T.dump_tensor(relay_ub, 2002, N, (1, N))

                T.copy(in_ub, B_direct[0, :])
                T.copy(relay_ub, B_relay[0, :])

    return main


def main():
    n = 513
    kernel = ub_copy_repro(n, "float32")

    a = torch.arange(n, dtype=torch.float32).reshape(1, n)
    a[0, -1] = 12345.0

    b_relay, b_direct = kernel(a)

    torch.npu.synchronize()

    print("input tail:", a[0, -8:].cpu())
    print("direct tail:", b_direct[0, -8:].cpu())
    print("relay tail:", b_relay[0, -8:].cpu())
    print("direct matches:", torch.equal(a.cpu(), b_direct.cpu()))
    print("relay matches:", torch.equal(a.cpu(), b_relay.cpu()))
    print("max abs diff direct:", (a.cpu() - b_direct.cpu()).abs().max().item())
    print("max abs diff relay:", (a.cpu() - b_relay.cpu()).abs().max().item())


if __name__ == "__main__":
    main()
