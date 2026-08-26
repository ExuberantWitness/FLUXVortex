# Formulation basis and claim boundary

The implemented operator follows three primary-source constraints:

1. Bucalem and Bathe's higher-order MITC16 construction supplies the fixed
   3x4, 4x3 and 3x3 covariant tying layouts and full 4x4 in-plane integration:
   <https://web.mit.edu/kjb/www/Publications_Prior_to_1998/Higher-Order_MITC_General_Shell_Elements.pdf>
2. Yamashita et al. motivate treating transverse shear with ANS and combining
   ANS with EAS for transverse-normal/thickness locking, including the
   center-configuration transformation:
   <https://iro.uiowa.edu/esploro/outputs/journalArticle/Continuum-Mechanics-Based-Bilinear-Shear-Deformable/9984196607202771>
3. Simo and Rifai provide the EAS orthogonality/patch requirements and the
   element-local static-condensation basis:
   <https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.1620290802>

The present code is a Q16 ANCF macro-shell realization of those constraints,
not a claim of line-by-line reproduction of any one paper. The evidence here
establishes rigid-motion consistency, constant-stress orthogonality, local
stationarity, energy/force/Jv consistency and CPU/CUDA parity. It does not yet
establish mesh-independent locking freedom, nonlinear time-integration
stability, or complete aeroelastic validation.

The local project library contained no corresponding PDF during this slice;
the formulation cross-check therefore used the linked primary sources and is
not a broader systematic literature review.
