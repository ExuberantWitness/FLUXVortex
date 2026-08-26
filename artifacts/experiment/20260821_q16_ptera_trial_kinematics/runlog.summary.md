# Run Log Summary

- Tests-first collection initially failed because the production adapter did
  not exist.
- The first implementation exposed a real upstream API boundary: Ptera's
  public `Wing.panels` setter is construction-only.  The fix rebuilds every
  Panel and atomically replaces the complete `_panels` owner only on the
  isolated pristine branch; individual Panel geometry/caches are not mutated.
- The first pilot used a motion too mild to cross the default LESP threshold.
  Rather than weaken the assertion, the bounded mechanism fixture freezes
  `lesp_crit=0.001` for both branches and obtains 24 real separated-LEV
  particles in each run.  This is an integration stimulus, not calibration.
- A topology negative initially failed at adapter construction, which is the
  correct earlier rejection.  The test was changed to a different topology
  having the same vertex count so the runtime panel-shape gate is exercised.
- The active-LEV transaction deliberately stops at the production unresolved
  impulse gate; no zero load or synthetic force distribution is substituted.
- Final focused and joint runs passed after Black formatting.  No paper data,
  GT or scorer was read.
