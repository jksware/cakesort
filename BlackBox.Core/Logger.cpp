#include "Logger.h"

#include <iostream>
#include <math.h>
#include <regex>

using namespace std;
using namespace BBox::Core;

void BBox::Core::Logger::BeginDocument()
{
	for (auto& pair : _writers)
		*pair.first << pair.second.BeginDocument << endl;
}

void BBox::Core::Logger::EndDocument()
{
	for (auto& pair : _writers)
		*pair.first << pair.second.EndDocument << endl;
}

void BBox::Core::Logger::BeginTable()
{
	_tableActive = true;
	for (auto& pair : _writers)
		*pair.first << pair.second.BeginTable << endl;
}

void BBox::Core::Logger::EndTable()
{
	_tableActive = false;
	for (auto& pair : _writers)
		*pair.first << pair.second.EndTable << endl;
}

void BBox::Core::Logger::BeginTabular()
{
	_tableActive = true;
	for (auto& pair : _writers)
		*pair.first << pair.second.BeginTabular << endl;
}

void BBox::Core::Logger::EndTabular()
{
	_tableActive = true;
	for (auto& pair : _writers)
		*pair.first << pair.second.EndTabular << endl;
}


void BBox::Core::Logger::ColumnHeader(string text)
{
	_headerActive = true;
	Log(text);
	_headerActive = false;
}

void BBox::Core::Logger::LogCaption(string text)
{
	for (auto& pair : _writers)
		*pair.first << (pair.second.BeginCaption + text + pair.second.EndCaption) << endl;
}

void BBox::Core::Logger::Flush()
{
	for (auto& pair : _writers)
		pair.first->flush();
}

//void Logger::Close() {
//	for each(auto pair in _writers)
//		pair.first->close();
//}

void BBox::Core::Logger::AddWriter(ostream* writer, const WriterOptions& options)
{
    pair<ostream*, WriterOptions> temp(writer, options);
	_writers.push_back(temp);
}

void BBox::Core::Logger::RemoveWriter(const ostream& writer)
{
	//_writers.erase();
}

void BBox::Core::Logger::Line()
{
	for (auto& pair : _writers) {
		if (pair.second.DoLinePad) {
			for (int i = 0; i < pair.second.PadLength; i++)
				*pair.first << pair.second.LinePad;
		}
		*pair.first << endl;
	}
}

//void BBox::Core::Logger::Log(string format, params object[] arg)
//{
//	string escaped = Regex.Escape(format);
//	for each(auto pair in _writers)
//	{
//		if (_headerActive)
//			pair.first.Write(pair.second.BeginColumnHeader);
//		pair.first.Write(pair.second.DoEscape ? escaped : format, arg);
//		if (_headerActive)
//			pair.first.Write(pair.second.EndColumnHeader);
//		if (!string.IsNullOrEmpty(pair.second.Separator))
//			pair.first.Write(pair.second.Separator);
//	}
//}

void BBox::Core::Logger::LogLine(string s)
{
	//string escaped = escape(format);
	string escaped = s;
	for (auto& pair : _writers) {
		if (_headerActive)
			*pair.first << pair.second.BeginColumnHeader;
		*pair.first << (pair.second.DoEscape ? escaped : s);
		if (_headerActive)
			*pair.first << pair.second.EndColumnHeader;
		*pair.first << (_tableActive ? pair.second.LineTerminator : string()) << endl;
	}
}

void BBox::Core::Logger::LogLine()
{
	for (auto& pair : _writers)
		*pair.first << (_tableActive ? pair.second.LineTerminator : string()) << endl;
}

void Logger::Log()
{
	for (auto& pair : _writers)
		*pair.first << pair.second.Separator;
}

void BBox::Core::Logger::Log(string s)
{
	for (auto& pair : _writers) {
		if (_headerActive)
			*pair.first << pair.second.BeginColumnHeader;
		*pair.first << s;
		if (_headerActive)
			*pair.first << pair.second.EndColumnHeader;
		*pair.first << pair.second.Separator;
	}
}

void BBox::Core::Logger::Log(float f)
{
	for (auto& pair : _writers) {
		//string tmp = float.IsPositiveInfinity(f) ? "+∞" : (pair.second.DoRoundDecimals ? Math.Round(f, pair.second.RoundDecimals) : f).ToString();
		//string tmp = pair.second.DoRoundDecimals ? round(f, pair.second.RoundDecimals) : f;

		*pair.first << pair.second.BeginNumber;
		*pair.first << f;
		*pair.first << pair.second.EndNumber;
		*pair.first << pair.second.Separator;
	}
}

void BBox::Core::Logger::Log(double f)
{
	for (auto& pair : _writers) {
		//string tmp = float.IsPositiveInfinity(f) ? "+∞" : (pair.second.DoRoundDecimals ? Math.Round(f, pair.second.RoundDecimals) : f).ToString();
		//string tmp = pair.second.DoRoundDecimals ? round(f, pair.second.RoundDecimals) : f;

		*pair.first << pair.second.BeginNumber;
		*pair.first << f;
		*pair.first << pair.second.EndNumber;
		*pair.first << pair.second.Separator;
	}
}

void BBox::Core::Logger::Log(int i)
{
	for (auto& pair : _writers) {
		*pair.first << pair.second.BeginNumber;
		*pair.first << i;
		*pair.first << pair.second.EndNumber;
		*pair.first << pair.second.Separator;
	}
}

void BBox::Core::Logger::Log(long l)
{
	for (auto& pair : _writers) {
		*pair.first << pair.second.BeginNumber;
		*pair.first << l;
		*pair.first << pair.second.EndNumber;
		*pair.first << pair.second.Separator;
	}
}
