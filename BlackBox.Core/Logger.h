#pragma once

#include <list>
#include <vector>
#include <iostream>
#include <memory>

#include "WriterOptions.h"

using namespace std;

namespace BBox
{
	namespace Core
	{
		class Logger
		{
			std::list <std::pair<std::ostream*, WriterOptions>> _writers;
			bool _tableActive;
			bool _headerActive;

		public:
			void BeginDocument();
			void EndDocument();
			void BeginTabular();
			void EndTabular();
			void BeginTable();
			void EndTable();
			void ColumnHeader(std::string name);
			void LogCaption(std::string text);
			void Flush();
			//void Close();
			void AddWriter(ostream* writer, const WriterOptions& writerOptions);
			void RemoveWriter(const std::ostream& writer);
			void Log();
			void LogLine(std::string s);

			template<typename T, typename ... Args>
			//typename enable_if<(sizeof...(Args) > 0), void>::type
			void
				LogLine(const string& format, T value, Args... args)
			{
				LogLine(format.data(), value, args...);
			}

			template<typename T, typename ... Args>
			typename enable_if<(sizeof...(Args) > 0), void>::type
				LogLine(const char *format, T value, Args ... args)
			{
				for (auto& pair : _writers) {
					const char *s = format;
					while (s && *s) {
						if (*s == '%' && *++s != '%') { // a for mat specifier (ignore which one it is)
							*pair.first << value;  // use first non-for mat argument
							LogLine(++s, args...);  // do a recursive call with the tail of the argument list
							return;
						}
						*pair.first << *s++;
					}
					throw std::runtime_error("extra arguments provided to LogLine");
				}
			}

			template<typename T, typename ... Args>
			typename enable_if<(sizeof...(Args) == 0), void>::type
				LogLine(const char *format, T value, Args ... args)
			{
				for (auto& pair : _writers) {
					const char *s = format;
					while (s && *s) {
						if (*s == '%' && *++s != '%') { // a for mat specifier (ignore which one it is)
							*pair.first << value;  // use first non-for mat argument
							LogLine(++s, args...);  // do a recursive call with the tail of the argument list
							*pair.first << (_tableActive ? pair.second.LineTerminator : string()) << endl;
							return;
						}
						*pair.first << *s++;
					}
					throw std::runtime_error("extra arguments provided to LogLine");
				}
			}

			void LogLine();
			void Log(std::string s);
			//void Log(std::string format, std::initializer_list<void*> arguments) = 0;
			void Log(float f);
			void Log(double f);
			void Log(int i);
			void Log(long l);
			void Line();
		};

	}
}