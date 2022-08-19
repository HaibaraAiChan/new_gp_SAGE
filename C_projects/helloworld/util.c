#include "util.h"

char * createUUID() {
	const char * c = "89ab";
	char * buf = (char *)malloc(37);
	char * p = buf;

	for (int n = 0; n < 16; ++n) {
		int b = rand() % 255;

		switch (n) {
			case 6: sprintf(p, "4%x", b % 15); break;
			case 8: sprintf(p, "%c%x", c[rand() % strlen(c)], b % 15); break;
			default: sprintf(p, "%02x", b); break;
		}

		p += 2;

		switch (n) {
			case 3:
			case 5:
			case 7:
			case 9:
				*p++ = '-'; break;
		}
	}
	*p = 0;
	return buf;
}

int getNumberLength(int number) {
	if (number < 0) return 0;

	if (number == 0) return 1;

	int length = 0;

	while (number != 0) {
		number /= 10;
		length++;
	}

	return length;
}

int gettimeofday(struct timeval *tp, void *tzp)
{
	time_t clock;
	struct tm tm;
	SYSTEMTIME wtm;
	GetLocalTime(&wtm);
	tm.tm_year = wtm.wYear - 1900;
	tm.tm_mon = wtm.wMonth - 1;
	tm.tm_mday = wtm.wDay;
	tm.tm_hour = wtm.wHour;
	tm.tm_min = wtm.wMinute;
	tm.tm_sec = wtm.wSecond;
	tm.tm_isdst = -1;
	clock = mktime(&tm);
	tp->tv_sec = clock;
	tp->tv_usec = wtm.wMilliseconds * 1000;
	return (0);
}