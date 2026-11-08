/*!
 * \file lib/gis/cpu_cores.c
 *
 * \brief GIS Library - Number of physical CPU cores.
 *
 * SPDX-FileCopyrightText: 2026 Vaclav Petras
 * SPDX-FileCopyrightText: GRASS Development Team
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* sched_getaffinity, CPU_ISSET */
#endif
#include <sched.h>
#endif
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include <grass/gis.h>

static int logical_cpus(void)
{
#if defined(__linux__)
    cpu_set_t set;

    if (sched_getaffinity(0, sizeof(set), &set) == 0)
        return CPU_COUNT(&set);
#endif
#if defined(_WIN32)
    SYSTEM_INFO info;

    GetSystemInfo(&info);
    return info.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_ONLN)
    long n = sysconf(_SC_NPROCESSORS_ONLN);

    if (n > 0)
        return (int)n;
#endif
    return 1;
}

#if defined(__linux__)
static int read_int(int cpu, const char *name)
{
    char path[128];
    FILE *fp;
    int value = -1;

    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s",
             cpu, name);
    fp = fopen(path, "r");
    if (!fp)
        return -1;
    if (fscanf(fp, "%d", &value) != 1)
        value = -1;
    fclose(fp);
    return value;
}

/* Count distinct (package, core) pairs among the CPUs this process may
 * run on, so that a CPU affinity mask (taskset, cgroup cpusets, batch
 * schedulers) limits the count like it limits the logical CPUs. */
static int linux_cores(void)
{
    cpu_set_t set;
    int *keys;
    int n = 0;

    if (sched_getaffinity(0, sizeof(set), &set) != 0)
        return 0;
    keys = G_malloc(2 * CPU_SETSIZE * sizeof(int));
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        int package, core, i;

        if (!CPU_ISSET(cpu, &set))
            continue;
        package = read_int(cpu, "physical_package_id");
        core = read_int(cpu, "core_id");
        if (package < 0 || core < 0) {
            n = 0;
            break;
        }
        for (i = 0; i < n; i++)
            if (keys[2 * i] == package && keys[2 * i + 1] == core)
                break;
        if (i == n) {
            keys[2 * n] = package;
            keys[2 * n + 1] = core;
            n++;
        }
    }
    G_free(keys);
    return n;
}
#endif

#if defined(_WIN32)
/* Counts the cores of the current processor group only (at most 64
 * logical processors) and ignores the affinity mask. */
static int windows_cores(void)
{
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *info;
    DWORD len = 0;
    int n = 0;

    GetLogicalProcessorInformation(NULL, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0)
        return 0;
    info = G_malloc(len);
    if (GetLogicalProcessorInformation(info, &len)) {
        DWORD count = len / sizeof(*info);

        for (DWORD i = 0; i < count; i++)
            if (info[i].Relationship == RelationProcessorCore)
                n++;
    }
    G_free(info);
    return n;
}
#endif

/*!
   \brief Get the number of physical CPU cores available to this process

   Logical CPUs of one core (SMT, Hyper-Threading) count once.

   On Linux, the count is limited to the CPUs in the affinity mask of the
   process. On macOS, it is the hw.physicalcpu sysctl value. On Windows,
   it is the number of cores reported by GetLogicalProcessorInformation.
   Where the topology is unknown, the number of logical CPUs is returned.

   \return number of physical cores (at least 1)

   \since version 8.6
 */
int G_available_cores(void)
{
    int n = 0;

#if defined(__linux__)
    n = linux_cores();
#elif defined(__APPLE__)
    int value;
    size_t size = sizeof(value);

    if (sysctlbyname("hw.physicalcpu", &value, &size, NULL, 0) == 0)
        n = value;
#elif defined(_WIN32)
    n = windows_cores();
#endif
    if (n < 1)
        n = logical_cpus();
    return n < 1 ? 1 : n;
}
