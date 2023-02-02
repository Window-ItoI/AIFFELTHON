# AIFFELTHON

> AIFFELTHON 대구 Window(ItoI)팀 프로젝트

## Member

> Window(ItoI) 팀 구성  

|이름|구성|역할|
|:---:|:---:|---|
|김용훈|팀장| 모델 검증 및 구조 파악, 모델 수정 및 Fine-tuning, 프로젝트 발표 PPT 제작, 발표|
|김태현|팀원| 데이터 선정 및 EDA, 데이터 전처리, 모델 수정 및 Fine-tuning,프로젝트 발표 PPT 제작|
|이진욱|팀원| 데이터 전처리, 프로젝트 발표 PPT 제작|

## 주제 소개

- 저희 주제는 베이스 모델인 Latent Diffusion을 활용한 Image Editing 입니다.
- 입력 값은 사용자가 원하는 배경과 지정 영역입니다. 
- 국내 랜드마크 DB를 통해 훈련한 모델이 지정한 영역 크기에 맞게 무작위 랜드마크 이미지를 배경과 자연스럽게 합성시켜 줍니다. 
- 추가로 무작위 이미지가 아닌 사용자가 원하는 랜드마크를 배경에 합성하는 기능까지 구현하는 것이 최종 목표입니다.

## 관련 논문

- [ ] [DDPM](https://arxiv.org/abs/2006.11239)
- [ ] [SDEdit](https://arxiv.org/pdf/2108.01073)
- [ ] [Latent Diffusion](https://arxiv.org/abs/2112.10752)

<details>
<summary> <b> Skip </b>  </summary>
<div markdown="1">

- [x] [AutoEncoder, Variational Auto Encoder](https://youtu.be/o_peo6U7IRM)
- [x] [DDIM](https://arxiv.org/abs/2010.02502)
- [x] [Score_SDE](https://arxiv.org/abs/2011.1345)
- [x] [DiffEdit](https://arxiv.org/abs/2210.11427)
- [x] [Blended Diffusion](https://arxiv.org/abs/2206.02779)

</div>
</details>
<br>

## 회의 기록

> Google Docs

- [Paper Study](https://docs.google.com/document/d/1EEC2o-0VEPsLeDhuRGIFZMoo0P9DC1aISvnzPlpCL3g/edit)

## 참고 Github

- [Awesome Diffusion Models](https://github.com/heejkoo/Awesome-Diffusion-Models)
- [Tutorials and Notebooks](https://github.com/heejkoo/Awesome-Diffusion-Models#tutorial-and-jupyter-notebook)
- [SDEdit](https://github.com/ermongroup/SDEdit)
- [Latent Diffusion](https://github.com/CompVis/latent-diffusion)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion#image-modification-with-stable-diffusion)
- [CLIP](https://github.com/openai/CLIP.git)


<details>
<summary> <b> Skip </b>  </summary>
<div markdown="1">

- [Score_SDE](https://github.com/yang-song/score_sde)
- [Image Editing](https://github.com/pfnet-research/multi-stage-blended-diffusion)
- [Semantic Image Translation](https://github.com/facebookresearch/SemanticImageTranslation)

</div>
</details>
<br>

## DataSet

- [ImageNet](https://www.image-net.org/)
  - [ImageNet Class List](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)
- [COCO](https://cocodataset.org/#home)
  - [COCO Category](https://eehoeskrap.tistory.com/368)
- [AI Hub Landmark Image](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=56)

## 개발 일정

|Task|기간|참조|
|---|---|---|
|논문 및 모델 공부|11월 10일 ~ 11월 18일|Diffusion 도메인 지식 공유|
|데이터 EDA|11월 14일 ~ 11월 15일|데이터 형태 및 사이즈 확인|
|데이터 전처리|11월 16일 ~ 11월 18일|학습데이터와 테스트 데이터셋 분류|
|Fine-tuning 및 모델 학습, 인퍼런스 구현|11월 21일 ~ 12월 7일|목적에 맞는 모델링 및 함수 구현|
|프로젝트 PPT 제작 및 발표|~ 12월 12일|프로젝트 최종 발표|
