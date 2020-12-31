# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import traitlets

from ..utils import Logger

from ..traits import trait_docs, Int, Unicode, Bool, List

from ..timing import function_timer

from .operator import Operator


@trait_docs
class TemplateMatrix(Operator):
    """Operator for projecting or accumulating template amplitudes."""

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    templates = List(
        None, allow_none=True, help="This should be a list of Template instances"
    )

    amplitudes = Unicode(None, allow_none=True, help="Data key for template amplitudes")

    transpose = Bool(False, help="If True, apply the transpose.")

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    det_data = Unicode(
        None, allow_none=True, help="Observation detdata key for the timestream data"
    )

    det_flags = Unicode(
        None, allow_none=True, help="Observation detdata key for flags to use"
    )

    det_flag_mask = Int(0, help="Bit mask value for optional flagging")

    shared_flags = Unicode(
        None, allow_none=True, help="Observation shared key for telescope flags to use"
    )

    shared_flag_mask = Int(0, help="Bit mask value for optional shared flagging")

    @traitlets.validate("templates")
    def _check_templates(self, proposal):
        temps = proposal["value"]
        if temps is None:
            return temps
        for tp in temps:
            if not isinstance(tp, Template):
                raise traitlets.TraitError(
                    "templates must be a list of Template instances or None"
                )
        return temps

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialized = False

    def apply_precond(self, amps_in, amps_out):
        """Apply the preconditioner from all templates to the amplitudes.

        This can only be called after the operator has been used at least once so that
        the templates are initialized.

        """
        if not self._initialized:
            raise RuntimeError(
                "You must call exec() once before applying preconditioners"
            )
        for tmpl in self.templates:
            tmpl.apply_precond(amps_in, amps_out)

    @function_timer
    def _exec(self, data, detectors=None, **kwargs):
        log = Logger.get()

        # Check that the detector data is set
        if self.det_data is None:
            raise RuntimeError("You must set the det_data trait before calling exec()")

        # Check that amplitudes is set
        if self.amplitudes is None:
            raise RuntimeError(
                "You must set the amplitudes trait before calling exec()"
            )

        # On the first call, we initialize all templates using the Data instance and
        # the fixed options for view, flagging, etc.
        if not self._initialized:
            for tmpl in self.templates:
                tmpl.view = self.view
                tmpl.det_flags = self.det_flags
                tmpl.det_flag_mask = self.det_flag_mask
                tmpl.shared_flags = self.shared_flags
                tmpl.shared_flag_mask = self.shared_flag_mask
                # This next line will trigger calculation of the number
                # of amplitudes within each template.
                tmpl.data = data
            self._initialized = True

        # Set the data we are using for this execution
        for tmpl in self.templates:
            tmpl.det_data = self.det_data

        if self.transpose:
            if self.amplitudes not in data:
                # The output template amplitudes do not yet exist.  Create these with
                # all zero values.
                data[self.amplitudes] = dict()
                for tmpl in self.templates:
                    data[self.amplitudes][tmpl.name] = tmpl.zeros()
            for ob in data.obs:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                for d in dets:
                    for tmpl in self.templates:
                        tmpl.project_signal(d, data[self.amplitudes[tmpl.name]])
        else:
            if self.amplitudes not in data:
                msg = "Template amplitudes '{}' do not exist in data".format(
                    self.amplitudes
                )
                log.error(msg)
                raise RuntimeError(msg)
            for ob in data.obs:
                # Get the detectors we are using for this observation
                dets = ob.select_local_detectors(detectors)
                if len(dets) == 0:
                    # Nothing to do for this observation
                    continue
                for d in dets:
                    for tmpl in self.templates:
                        tmpl.add_to_signal(d, data[self.amplitudes[tmpl.name]])
        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": list(),
            "detdata": list(),
            "intervals": list(),
        }
        if self.view is not None:
            req["intervals"].append(self.view)
        if self.transpose:
            req["detdata"].append(self.det_data)
            if self.shared_flags is not None:
                req["shared"].append(self.shared_flags)
            if self.det_flags is not None:
                req["detdata"].append(self.det_flags)
        return req

    def _provides(self):
        prov = dict()
        if not self.transpose:
            prov["detdata"] = [self.det_data]
        return prov

    def _accelerators(self):
        return list()