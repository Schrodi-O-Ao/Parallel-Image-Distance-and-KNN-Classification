#ifndef PARALLEL_UTILS_HPP
#define PARALLEL_UTILS_HPP

/**
 * @file parallel_utils.hpp
 * @brief Thread synchronization primitives, thread pool, and parallel reduction utilities.
 * @author ST20290339
 */

#include "common.hpp"
#include <queue>

namespace cis6007 {
namespace parallel_utils {

/**
 * @class Barrier
 * @brief Barrier synchronization primitive for thread coordination across computational phases.
 */
class Barrier {
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t count_;          // Remaining threads yet to arrive
    size_t initial_count_;  // Original thread count for reset
    size_t generation_;     // Incremented each time barrier is released

public:
    /**
     * @brief Construct a barrier for the specified number of threads.
     *
     * @param thread_count Number of threads that will wait at the barrier.
     */
    explicit Barrier(size_t thread_count)
        : count_(thread_count), initial_count_(thread_count), generation_(0) {}

    /**
     * @brief Wait at the barrier until all participating threads have arrived.
     *
     * Reusable: resets automatically after all threads arrive, advancing the generation.
     *
     * @return true for the last arriving thread (the leader), false for all other threads.
     * @warning Reusing a barrier requires all threads from the previous phase to exit wait()
     *          before any thread enters the subsequent wait() phase.
     */
    bool wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        size_t gen = generation_;

        if (--count_ == 0) {
            // Last thread to arrive: reset for next use and wake everyone
            generation_++;
            count_ = initial_count_;  // Restore to original thread count
            cv_.notify_all();
            return true;  // This thread is the "leader"
        }

        // All other threads wait until the generation changes
        cv_.wait(lock, [this, gen] { return generation_ != gen; });
        return false;
    }
};

/**
 * @class ParallelReduction
 * @brief Parallel reduction operations combining vector elements across threads.
 * @tparam T Value type to reduce (must support addition, comparison, or copy construction).
 */
template<typename T>
class ParallelReduction {
public:
    /**
     * @brief Reduce a vector of values using the sum operation in parallel.
     *
     * @param values Vector of values to reduce.
     * @param num_threads Number of worker threads for parallel reduction.
     * @return Sum of all elements in values.
     */
    static T sum(const std::vector<T>& values, int num_threads = std::thread::hardware_concurrency()) {
        if (values.empty()) return T(0);
        
        T total = T(0);
        std::mutex mutex;
        std::vector<std::thread> threads;
        
        auto worker = [&](size_t start, size_t end) {
            T local_sum = T(0);
            for (size_t i = start; i < end; ++i) {
                local_sum += values[i];
            }
            std::lock_guard<std::mutex> lock(mutex);
            total += local_sum;
        };
        
        size_t chunk_size = values.size() / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = (t == num_threads - 1) ? values.size() : (t + 1) * chunk_size;
            threads.emplace_back(worker, start, end);
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        return total;
    }
    
    /**
     * @brief Reduce a vector of values using the min operation in parallel.
     *
     * @param values Vector of values to reduce.
     * @param num_threads Number of worker threads for parallel reduction.
     * @return Minimum element in values.
     */
    static T min(const std::vector<T>& values, int num_threads = std::thread::hardware_concurrency()) {
        if (values.empty()) return T(0);
        
        T minimum = values[0];
        std::mutex mutex;
        std::vector<std::thread> threads;
        
        auto worker = [&](size_t start, size_t end) {
            T local_min = values[start];
            for (size_t i = start + 1; i < end; ++i) {
                if (values[i] < local_min) {
                    local_min = values[i];
                }
            }
            std::lock_guard<std::mutex> lock(mutex);
            if (local_min < minimum) {
                minimum = local_min;
            }
        };
        
        size_t chunk_size = values.size() / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = (t == num_threads - 1) ? values.size() : (t + 1) * chunk_size;
            threads.emplace_back(worker, start, end);
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        return minimum;
    }
    
    /**
     * @brief Reduce a vector of values using the max operation in parallel.
     *
     * @param values Vector of values to reduce.
     * @param num_threads Number of worker threads for parallel reduction.
     * @return Maximum element in values.
     */
    static T max(const std::vector<T>& values, int num_threads = std::thread::hardware_concurrency()) {
        if (values.empty()) return T(0);
        
        T maximum = values[0];
        std::mutex mutex;
        std::vector<std::thread> threads;
        
        auto worker = [&](size_t start, size_t end) {
            T local_max = values[start];
            for (size_t i = start + 1; i < end; ++i) {
                if (values[i] > local_max) {
                    local_max = values[i];
                }
            }
            std::lock_guard<std::mutex> lock(mutex);
            if (local_max > maximum) {
                maximum = local_max;
            }
        };
        
        size_t chunk_size = values.size() / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = (t == num_threads - 1) ? values.size() : (t + 1) * chunk_size;
            threads.emplace_back(worker, start, end);
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        return maximum;
    }
};

/**
 * @class ThreadPool
 * @brief Thread pool managing worker threads to amortize thread creation overhead.
 */
class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;

public:
    /**
     * @brief Construct a thread pool with the specified number of worker threads.
     *
     * @param num_threads Number of worker threads to instantiate.
     */
    explicit ThreadPool(size_t num_threads) : stop_(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }
    
    /**
     * @brief Destructor: signals termination and joins all worker threads.
     */
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }
    
    /**
     * @brief Enqueue a callable task for execution by an available worker thread.
     *
     * @tparam F Callable type taking no arguments and returning void.
     * @param f Function or lambda task to execute.
     * @warning Must not be called after ThreadPool shutdown has initiated.
     */
    template<typename F>
    void enqueue(F f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(f);
        }
        condition_.notify_one();
    }
};

/**
 * @brief Get optimal thread count tailored to workload size and hardware concurrency.
 *
 * @param workload_size Size of the workload (e.g., number of elements).
 * @param min_threads Minimum number of threads to return.
 * @return Optimal thread count.
 */
int get_optimal_thread_count(size_t workload_size, int min_threads = 1);

/**
 * @brief Apply a transform function to each element of a vector in parallel.
 *
 * @tparam T Input element type.
 * @tparam F Callable type taking const T& and returning a result value.
 * @param input Input vector to process.
 * @param func Transformation function applied to each element.
 * @param num_threads Worker thread count (default: hardware concurrency).
 * @return Vector of transformed results matching the order and size of input.
 * @note Uses fixed-stride chunking across threads.
 */
template<typename T, typename F>
std::vector<typename std::invoke_result_t<F, T>> parallel_map(
    const std::vector<T>& input,
    F func,
    int num_threads = std::thread::hardware_concurrency()
) {
    using ResultType = typename std::invoke_result_t<F, T>;
    std::vector<ResultType> output(input.size());
    
    std::vector<std::thread> threads;
    size_t chunk_size = input.size() / num_threads;
    
    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            output[i] = func(input[i]);
        }
    };
    
    for (int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? input.size() : (t + 1) * chunk_size;
        threads.emplace_back(worker, start, end);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return output;
}

/**
 * MPI initialization and cleanup helper
 */
#ifdef USE_MPI
class MPIHelper {
public:
    /**
     * Initialize MPI environment
     * Should be called once at program start
     */
    static void initialize(int& argc, char**& argv) {
        MPI_Init(&argc, &argv);
    }
    
    /**
     * Finalize MPI environment
     * Should be called once at program end
     */
    static void finalize() {
        MPI_Finalize();
    }
    
    /**
     * Get MPI rank of current process
     */
    static int get_rank() {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }
    
    /**
     * Get total number of MPI processes
     */
    static int get_size() {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }
    
    /**
     * Barrier synchronization across all MPI processes
     */
    static void barrier() {
        MPI_Barrier(MPI_COMM_WORLD);
    }
};
#endif

} // namespace parallel_utils
} // namespace cis6007

#endif // PARALLEL_UTILS_HPP
