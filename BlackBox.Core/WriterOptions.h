#pragma once

#include <string>

namespace BBox {
	namespace Core {

		struct WriterOptions
		{
			/* regular stuff */
			std::string Separator;
			std::string LineTerminator;
			bool DoEscape;

			/* line padding stuff */
			bool DoLinePad;
			std::string LinePad;
			int PadLength;

			/* float arithmetic rounding stuff */
			bool DoRoundDecimals;
			int RoundDecimals;

			/* environment stuff */
			std::string BeginColumnHeader;
			std::string EndColumnHeader;
			std::string BeginNumber;
			std::string EndNumber;
			std::string BeginDocument;
			std::string EndDocument;
			std::string BeginTable;
			std::string EndTable;
			std::string BeginTabular;
			std::string EndTabular;
			std::string BeginCaption;
			std::string EndCaption;
		};
	}
}