// CPU-only regression of the production inventory/runtime identity. No model or GPU is loaded.
#include "ds41_persistent_identity.hpp"
#include <cstdlib>
#include <iostream>
#ifndef _WIN32
#include <unistd.h>
#endif
using namespace strata::v41;
static void check(bool ok, const char * message) {
    if (!ok) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
int main() {
#ifdef __linux__
    char directory[] = "/tmp/strata-ds41-identity-XXXXXX";
    check(mkdtemp(directory), "private fixture directory");
    namespace fs = std::filesystem;
    const fs::path root(directory);
    const auto main1 = root / "main-00001-of-00002.gguf", main2 = root / "main-00002-of-00002.gguf";
    const auto draft1 = root / "draft-00001-of-00002.gguf", draft2 = root / "draft-00002-of-00002.gguf";
    auto write = [](const fs::path & path, const std::string & value) { std::ofstream out(path); out << value; };
    write(main1,"main metadata"); write(main2,"main weights A");
    write(draft1,"draft metadata"); write(draft2,"draft weights A");
    char exe[]="ds41_generate", model[]="--model", path[]="stable-model-path";
    char * argv[]{exe,model,path};
    std::vector<std::string> paths{main1.string(),draft1.string()};
    const std::vector<std::string> env{"STRATA_TEST_ARITHMETIC=1"};
    int queries=0;
    std::string runtime="backend=cuda;device=0000:01:00.0;arch=120;runtime=13010;driver=13010";
    auto query = [&] { ++queries; return runtime; };
    auto identity = [&] { return ds41_persistent_identity_if_enabled(true,3,argv,paths,env,query); };
    check(ds41_persistent_identity_if_enabled(false,3,argv,paths,env,query).empty() && queries==0,
          "disabled tier must not query backend");
    const auto original=identity();
    check(!original.empty() && identity()==original,"unchanged resolved model and runtime are stable");
    write(main2,"main weights B");
    const auto changed_main=identity();
    check(!changed_main.empty() && changed_main!=original,"later main shard mutation invalidates identity");
    std::ifstream first(main1); std::string first_bytes((std::istreambuf_iterator<char>(first)),{});
    check(first_bytes=="main metadata","first shard stayed unchanged");
    write(draft2,"draft weights B");
    const auto changed_draft=identity();
    check(!changed_draft.empty() && changed_draft!=changed_main,"later DSpark shard mutation invalidates identity");
    const auto original_runtime=runtime;
    for (const auto & change : std::vector<std::pair<std::string,std::string>>{
            {"backend=cuda","backend=cpu"}, {"device=0000:01:00.0","device=0000:02:00.0"},
            {"arch=120","arch=90"}, {"runtime=13010","runtime=13020"}, {"driver=13010","driver=13020"}}) {
        runtime=original_runtime;
        runtime.replace(runtime.find(change.first),change.first.size(),change.second);
        const auto changed=identity();
        check(!changed.empty() && changed!=changed_draft,"runtime/backend/device/architecture changes invalidate identity");
    }
    runtime.clear(); check(identity().empty(),"failed runtime query disables persistence");
    check(ds41_persistent_identity_if_enabled(true,3,argv,paths,env,[]()->std::string{throw 1;}).empty(),
          "query exception is a safe miss");
    runtime="valid backend descriptor";
    fs::remove(main2); check(identity().empty(),"missing later main shard disables persistence");
    write(main2,"main weights B"); fs::remove(draft2);
    check(identity().empty(),"missing later draft shard disables persistence");
    fs::remove_all(root); // only this test's disposable files
    std::cout << "DS41 persistent runtime/split inventory CPU regression PASS (no GPU queries)\n";
#endif
}
