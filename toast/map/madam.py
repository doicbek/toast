# Copyright (c) 2015 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by 
# a BSD-style license that can be found in the LICENSE file.


import os
if 'TOAST_NO_MPI' in os.environ.keys():
    from .. import fakempi as MPI
else:
    from mpi4py import MPI

import ctypes as ct
from ctypes.util import find_library

import unittest

import numpy as np
import numpy.ctypeslib as npc

import healpy as hp

from ..dist import Comm, Data
from ..operator import Operator
from ..tod import TOD
from ..tod import Interval

libmadam = None
if 'TOAST_NO_MPI' not in os.environ.keys():
    try:
        libmadam = ct.CDLL('libmadam.so')
    except:
        path = find_library('madam')
        if path is not None:
            libmadam = ct.CDLL(path)

if libmadam is not None:
    libmadam.destripe.restype = None
    libmadam.destripe.argtypes = [
        ct.c_int,
        ct.c_char_p,
        ct.c_long,
        ct.c_char_p,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ct.c_long,
        ct.c_long,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ct.c_long,
        npc.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
        ct.c_long,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ct.c_long,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ct.c_long,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    libmadam.destripe_with_cache.restype = None
    libmadam.destripe_with_cache.argtypes = [
        ct.c_int,
        ct.c_long,
        ct.c_long,
        ct.c_long,
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npc.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ]
    libmadam.clear_caches.restype = None
    libmadam.clear_caches.argtypes = []

# Some keys may be defined multiple times in the Madam parameter files.
# Assume that such entries are aggregated into a list in a parameter
# dictionary

repeated_keys = ['detset', 'detset_nopol', 'survey']


class OpMadam(Operator):
    """
    Operator which passes data to libmadam for map-making.

    Args:
        params (dictionary): parameters to pass to madam.
        detweights (dictionary): individual noise weights to use for each
            detector.
        pixels (str): the name of the cache object (<pixels>_<detector>)
            containing the pixel indices to use.
        pixels_nested (bool): Set to False if the pixel numbers are in
            ring ordering. Default is True.
        weights (str): the name of the cache object (<weights>_<detector>)
            containing the pointing weights to use.
        name (str): the name of the cache object (<name>_<detector>) to
            use for the detector timestream.  If None, use the TOD.
        name_out (str): the name of the cache object (<name>_<detector>)
            to use to output destriped detector timestream.
            No output if None.
        flag_name (str): the name of the cache object
            (<flag_name>_<detector>) to use for the detector flags.
            If None, use the TOD.
        flag_mask (int): the integer bit mask (0-255) that should be 
            used with the detector flags in a bitwise AND.
        common_flag_name (str): the name of the cache object 
            (<common_flag_name>_<detector>) to use for the common flags.  
            If None, use the TOD.
        common_flag_mask (int): the integer bit mask (0-255) that should
            be used with the common flags in a bitwise AND.
        apply_flags (bool): whether to apply flags to the pixel numbers.
        timestamps_name (str): the name of the cache object to use for
            time stamps.
        purge (bool): if True, clear any cached data that is copied into
            the Madam buffers.
        dets (iterable):  List of detectors to map. If left as None, all
            available detectors are mapped.
        mcmode (bool): If true, the operator is constructed in
             Monte Carlo mode and Madam will cache auxiliary information
             such as pixel matrices and noise filter.
        noise (str):  Keyword to use when retrieving the noise object
             from the observation.
    """

    def __init__(self, params={}, timestamps_name=None, detweights=None,
                 pixels='pixels', pixels_nested=True, weights='weights',
                 name=None, name_out=None, flag_name=None, flag_mask=255,
                 common_flag_name=None, common_flag_mask=255,
                 apply_flags=True, purge=False, dets=None, mcmode=False,
                 noise='noise'):
        
        # We call the parent class constructor, which currently does nothing
        super().__init__()
        # madam uses time-based distribution
        self._timedist = True
        self._timestamps_name = timestamps_name
        self._name = name
        self._name_out = name_out
        self._flag_name = flag_name
        self._flag_mask = flag_mask
        self._common_flag_name = common_flag_name
        self._common_flag_mask = common_flag_mask
        self._pixels = pixels
        self._pixels_nested = pixels_nested
        self._weights = weights
        self._detw = detweights
        self._purge = purge
        self._apply_flags = apply_flags
        self._params = params
        if dets is not None:
            self._dets = set( dets )
        else:
            self._dets = None
        self._mcmode = mcmode
        if mcmode:
            self._params['mcmode'] = True
        else:
            self._params['mcmode'] = False
        self._cached = False
        self._noisekey = noise

    def __del__(self):
        if self._cached:
            libmadam.clear_caches()
            self._cached = False

    @property
    def timedist(self):
        """
        (bool): Whether this operator requires data that time-distributed.
        """
        return self._timedist

    @property
    def available(self):
        """
        (bool): True if libmadam is found in the library search path.
        """
        return (libmadam is not None)

    def _dict2parstring(self, d):
        s = ''
        for key, value in d.items():
            if key in repeated_keys:
                for separate_value in value:
                    s += '{} = {};'.format(key, separate_value)
            else:
                s += '{} = {};'.format(key, value)
        return s

    def _dets2detstring(self, dets):
        s = ''
        for d in dets:
            s += '{};'.format(d)
        return s

    def exec(self, data):
        """
        Copy data to Madam-compatible buffers and make a map.

        Args:
            data (toast.Data): The distributed data.
        """
        if libmadam is None:
            raise RuntimeError("Cannot find libmadam")

        # the two-level pytoast communicator
        comm = data.comm

        # Madam only works with a data model where there is either
        #   1) one global observation split among the processes,
        #      and where the data is distributed time-wise OR
        #   2) one or more private observation per process

        if len(data.obs) != 1:
            nsamp = 0
            for obs in data.obs:
                tod = obs['tod']
                if tod.mpicomm.size != 1:
                    raise RuntimeError(
                        'When calling Madam, TOD must either be globally '
                        'distributed or each observation assigned to exactly '
                        'one task')
                nsamp += tod.local_samples[1]
        else:
            tod = data.obs[0]['tod']
            if not tod.timedist:
                raise RuntimeError(
                    "Madam requires data to be distributed by time")
            nsamp = tod.local_samples[1]

        # Determine the detectors and the pointing matrix non-zeros
        # from the first observation. Madam will expect these to remain
        # unchanged across observations.

        tod = data.obs[0]['tod']

        if self._dets is None:
            detectors = tod.detectors
        else:
            detectors = [det for det in tod.detectors
                         if det in self._dets]

        detstring = self._dets2detstring(detectors)

        # to get the number of Non-zero pointing weights per pixel,
        # we use the fact that for Madam, all processes have all detectors
        # for some slice of time.  So we can get this information from the
        # shape of the data from the first detector

        nnzname = "{}_{}".format(self._weights, detectors[0])
        nnz_full = tod.cache.reference(nnzname).shape[1]

        if 'temperature_only' in self._params \
           and self._params['temperature_only'] in [
               'T', 'True', 'TRUE', 'true', True]:
            if nnz_full not in [1, 3]:
                raise RuntimeError(
                    'OpMadam: Don\'t know how to make a temperature map '
                    'with nnz={}'.format(nnz_full))
            nnz = 1
            nnz_stride = nnz_full
        else:
            nnz = nnz_full
            nnz_stride = 1

        fcomm = comm.comm_world.py2f()

        if 'nside_map' not in self._params:
            raise RuntimeError(
                'OpMadam: "nside_map" must be set in the parameter dictionary')
        nside = self._params['nside_map']

        parstring = self._dict2parstring(self._params)

        # create madam-compatible buffers

        ndet = len(detectors)

        madam_timestamps = []
        madam_signal = []
        madam_pixels = []
        madam_pixweights = []
        for d in range(ndet):
            madam_signal.append([])
            madam_pixels.append([])
            madam_pixweights.append([])

        offset = 0
        period_lengths = []
        obs_period_ranges = []

        psdfreqs = None
        psds = {}

        for obs in data.obs:
            tod = obs['tod']
            nlocal = tod.local_samples[1]
            period_ranges = []

            # get the total list of intervals
            intervals = None
            if 'intervals' in data.obs[0].keys():
                intervals = data.obs[0]['intervals']
            if intervals is None:
                intervals = [Interval(start=0.0, stop=0.0, first=0,
                                      last=(tod.total_samples-1))]

            local_offset = tod.local_samples[0]
            local_nsamp = tod.local_samples[1]
            for ival in intervals:
                if (ival.last >= local_offset
                    and ival.first < (local_offset + local_nsamp)):
                    local_start = ival.first - local_offset
                    local_stop = ival.last - local_offset
                    if local_start < 0:
                        local_start = 0
                    if local_stop > local_nsamp:
                        local_stop = local_nsamp
                    period_lengths.append(local_stop - local_start)
                    period_ranges.append((local_start, local_stop))
            obs_period_ranges.append(period_ranges)

            # Collect the timestamps for the valid intervals
            if self._timestamps_name is not None:
                timestamps = tod.cache.reference(self._timestamps_name)
            else:
                timestamps = tod.read_times()

            local_offset = offset
            for istart, istop in period_ranges:
                nn = istop - istart
                dslice = slice(d*nsamp + local_offset,
                               d*nsamp + local_offset + nn)
                madam_timestamps.append(timestamps[istart:istop])
                local_offset += nn

            # get the noise object for this observation and create new
            # entries in the dictionary when the PSD actually changes
            if self._noisekey in obs.keys():
                nse = obs[self._noisekey]
                if psdfreqs is None:
                    psdfreqs = nse.freq(detectors[0]).astype(np.float64).copy()
                    npsdbin = len(psdfreqs)
                for d in range(ndet):
                    det = detectors[d]
                    check_psdfreqs = nse.freq(det)
                    if not np.allclose(psdfreqs, check_psdfreqs):
                        raise RuntimeError('All PSDs passed to Madam must have'
                                           ' the same frequency binning.')
                    psd = nse.psd(det)
                    if det not in psds:
                        psds[det] = [(0, psd)]
                    else:
                        if not np.allclose(psds[det][-1][1], psd):
                            psds[det] += [(timestamps[0], psd)]

            for d in range(ndet):
                # Get the signal.

                cachename = None
                if self._name is not None:
                    cachename = "{}_{}".format(self._name, detectors[d])
                    signal = tod.cache.reference(cachename)
                else:
                    signal = tod.read(detector=detectors[d])

                # Optionally get the flags, otherwise they are assumed to
                # have been applied to the pixel numbers.

                if self._flag_name is not None:
                    cacheflagname = "{}_{}".format(
                        self._flag_name, detectors[d])

                if self._apply_flags:

                    if self._flag_name is not None:
                        detflags = tod.cache.reference(cacheflagname)
                        flags = (detflags & self._flag_mask) != 0
                        if self._common_flag_name is not None:
                            commonflags = tod.cache.reference(
                                self._common_flag_name)
                            flags[(commonflags & self._common_flag_mask)
                                  != 0] = True
                    else:
                        detflags, commonflags = tod.read_flags(
                            detector=detectors[d])
                        flags = np.logical_or(
                            (detflags & self._flag_mask) != 0,
                            (commonflags & self._common_flag_mask) != 0)

                # get the pixels and weights for the valid intervals from the cache

                pixelsname = "{}_{}".format(self._pixels, detectors[d])
                weightsname = "{}_{}".format(self._weights, detectors[d])
                pixels = tod.cache.reference(pixelsname)
                weights = tod.cache.reference(weightsname)

                if not self._pixels_nested:
                    # Madam expects the pixels to be in nested ordering
                    pixels = pixels.copy()
                    pixels = hp.ring2nest( nside, pixels )

                if self._apply_flags:
                    pixels = pixels.copy()
                    pixels[flags] = -1

                local_offset = offset
                for istart, istop in period_ranges:
                    nn = istop - istart
                    dslice = slice(d*nsamp + local_offset,
                                   d*nsamp + local_offset + nn)
                    dwslice = slice((d*nsamp+local_offset) * nnz,
                                    (d*nsamp+local_offset+nn) * nnz)
                    madam_signal[d].append(signal[istart:istop])
                    madam_pixels[d].append(pixels[istart:istop].copy())
                    madam_pixweights[d].append(
                        weights[istart:istop].flatten()[::nnz_stride].copy())

                    local_offset += nn

                if self._purge:
                    tod.cache.clear(pattern=pixelsname)
                    tod.cache.clear(pattern=weightsname)
                    if self._name is not None:
                        tod.cache.clear(pattern=cachename)
                    if self._flag_name is not None:
                        tod.cache.clear(pattern=cacheflagname)

            if self._purge:
                if self._common_flag_name is not None:
                    tod.cache.clear(pattern=self._common_flag_name)

            offset = local_offset

        madam_timestamps = np.hstack(madam_timestamps).astype(np.float64)
        for d in range(ndet):
            madam_signal[d] = np.hstack(madam_signal[d])
            madam_pixels[d] = np.hstack(madam_pixels[d])
            madam_pixweights[d] = np.hstack(madam_pixweights[d])
        madam_signal = np.hstack(madam_signal).astype(np.float64)
        madam_pixels = np.hstack(madam_pixels).astype(np.int64)
        madam_pixweights = np.hstack(madam_pixweights).astype(np.float64)

        nperiod = len(period_lengths)
        period_lengths = np.array(period_lengths, dtype=np.int64)
        nsamp = np.sum(period_lengths, dtype=np.int64)
        periods = np.zeros(nperiod, dtype=np.int64)
        for i, n in enumerate(period_lengths[:-1]):
            periods[i+1] = periods[i] + n

        if self._cached:
            # destripe
            libmadam.destripe_with_cache(
                fcomm, ndet, nsamp, nnz, madam_timestamps, madam_pixels,
                madam_pixweights, madam_signal)
        else:
            # detweights is either a dictionary of weights specified at
            # construction time, or else we use uniform weighting.
            detw = {}
            if self._detw is None:
                for d in range(ndet):
                    detw[detectors[d]] = 1.0
            else:
                detw = self._detw

            detweights = np.zeros(ndet, dtype=np.float64)
            for d in range(ndet):
                detweights[d] = detw[detectors[d]]

            if len(psds) > 0:
                npsdbin = len(psdfreqs)

                npsd = np.zeros(ndet, dtype=np.int64)
                psdstarts = []
                psdvals = []
                for d in range(ndet):
                    det = detectors[d]
                    if det not in psds:
                        raise RuntimeError('Every detector must have at least '
                                           'one PSD')
                    psdlist = psds[det]
                    npsd[d] = len(psdlist)
                    for psdstart, psd in psdlist:
                        psdstarts.append(psdstart)
                        psdvals.append(psd)
                npsdtot = np.sum(npsd)
                psdstarts = np.array(psdstarts, dtype=np.float64)
                psdvals = np.hstack(psdvals).astype(np.float64)
                npsdval = psdvals.size
            else:
                npsd = np.ones(ndet, dtype=np.int64)
                npsdtot = np.sum(npsd)
                psdstarts = np.zeros(npsdtot)
                npsdbin = 10
                fsample = 10.
                psdfreqs = np.arange(npsdbin) * fsample / npsdbin
                npsdval = npsdbin * npsdtot
                psdvals = np.ones(npsdval)

            # destripe

            libmadam.destripe(
                fcomm, parstring.encode(), ndet, detstring.encode(), detweights,
                nsamp, nnz, madam_timestamps, madam_pixels, madam_pixweights,
                madam_signal, nperiod, periods, npsd, npsdtot, psdstarts,
                npsdbin, psdfreqs, npsdval, psdvals)

            if self._mcmode:
                self._cached = True

        if self._name_out is not None:
            for obs, period_ranges in zip(data.obs, obs_period_ranges):
                tod = obs['tod']
                nlocal = tod.local_samples[1]
                for d, det in enumerate(detectors):
                    signal = np.ones(nlocal) * np.nan
                    offset = 0
                    for istart, istop in period_ranges:
                        nn = istop - istart
                        signal[istart:istop] = madam_signal[offset:offset+nn]
                        offset += nn
                    cachename = "{}_{}".format(self._name_out, det)
                    tod.cache.put(cachename, signal, replace=True)
                offset += nlocal

        return
