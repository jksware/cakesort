#pragma once

#include <chrono>
#include <time.h>

#ifdef _WIN32
#define INLINE __forceinline
#include <Windows.h>

#else
#define INLINE inline

#endif

namespace BBox
{
	namespace Core
	{

		class Stopwatch
		{
			typedef std::chrono::high_resolution_clock my_clock;
			typedef std::chrono::time_point<my_clock> time;

			bool _running;
			time _begin;
			my_clock::duration _elapsed;

		public:
			typedef my_clock::duration duration;

			Stopwatch() : _running(false)
			{
				Reset();
			}

			INLINE bool IsRunning()
			{
				return _running;
			}

			INLINE void Start()
			{
				if (_running)
					return;

				_running = true;
				_begin = my_clock::now();
			}

			INLINE void Restart()
			{
				Reset();
				_running = true;
			}

			INLINE void Reset()
			{
				_elapsed = my_clock::duration::zero();
				_begin = my_clock::now();
			}

			INLINE void Stop()
			{
				if (!_running)
					return;

				_elapsed += my_clock::now() - _begin;
				_running = false;
			}

			INLINE my_clock::duration Elapsed()
			{
				return _elapsed + (_running ? (my_clock::now() - _begin) : my_clock::duration::zero());
			}

			template <typename duration_type>
			INLINE duration_type ElapsedMiliseconds()
			{
				return std::chrono::duration_cast<std::chrono::duration<duration_type, std::milli>>(Elapsed()).count();
			}

			template <typename duration_type>
			INLINE duration_type ElapsedSeconds()
			{
				return std::chrono::duration_cast<std::chrono::duration<duration_type, std::ratio<1, 1>>>(Elapsed()).count();
			}

			template <typename duration_type>
			static duration_type Frequency()
			{
#ifdef _WIN32
				long long freq;
				QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
				return static_cast<duration_type>(freq);
#else
				struct timespec tp;
				clock_getres(CLOCK_MONOTONIC, &tp);
				auto freq = 1e9 * ((unsigned long long) tp.tv_sec * 1E9L + (unsigned long long) tp.tv_nsec);
				return static_cast<duration_type>(freq);
#endif
			}
		};
	}
}
