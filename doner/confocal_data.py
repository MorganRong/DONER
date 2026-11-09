NOISE_REVISION='confocal_text_operator_v1'
class DomainData:
    def __init__(self,native):
        self.native=native

    def __getattr__(self,name):
        return getattr(self.native,name)

    def get_batch(self,step,device='cpu'):
        batch=self.native.get_batch(step,device)
        clean_step=bool(batch['audit']['clean_step'])
        families=[]
        names=[]
        for m in batch['metadata']:
            if clean_step:
                m.update(name='clean',family='clean',style_id='confocal|clean',signature='clean')
            name=m.get('name') or m.get('family') or m.get('signature')
            family=m.get('family') or name
            if not isinstance(name,str) or not name or not isinstance(family,str) or not family:
                raise ValueError('A real confocal operator identity is required')
            if not clean_step and (not m.get('signature') or not m.get('style_id')):
                raise ValueError('Missing contrastive operator labels')
            m.update(domain='hagen',image_domain='hagen',operator_bank='confocal_text',
                     name=name,family=family,anchor_group=family,anchor_name=name,
                     noise_revision=NOISE_REVISION,
                     realization_RMS_normalized=not clean_step,
                     selected_HQ_used_for_parent_noise_arithmetic=False,
                     noise_conditioned_on_actual_augmented_clean_target=True,
                     full_noise_extracted_as_parent_float32_noisy_minus_clean=True,
                     legacy_STM_operator_used=False)
            families.append(family)
            names.append(name)

        batch['audit'].update(anchor_group=families[0],anchor_name=names[0],
                              operator_families=families,operator_names=names,
                              noise_domain='hagen',noise_revision=NOISE_REVISION,
                              legacy_STM_operator_used=False)
        return batch
