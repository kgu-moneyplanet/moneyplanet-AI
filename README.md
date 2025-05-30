# BE
⚙️ BackEnd 코드

## Branch

| **머릿말** | **설명** |
| --- | --- |
| main | 서비스 브랜치 |
| release | 배포 전 작업 기준 |
| feature | 기능 단위 구현 |
| hotfix | 서비스 중 긴급 수정 건에 대한 처리 |
| BE | 백엔드 개발 부분 |

## 🤝 Commit Convention

| 머릿말           | 설명                                                                      |
| ---------------- | ------------------------------------------------------------------------- |
| feat             | 새로운 기능 추가                                                          |
| fix              | 버그 수정                                                                 |
| design           | CSS 등 사용자 UI 디자인 변경                                              |
| !BREAKING CHANGE | 커다란 API 변경의 경우                                                    |
| !HOTFIX          | 코드 포맷 변경, 세미 콜론 누락, 코드 수정이 없는 경우                     |
| refactor         | 프로덕션 코드 리팩토링업                                                  |
| comment          | 필요한 주석 추가 및 변경                                                  |
| docs             | 문서 수정                                                                 |
| test             | 테스트 추가, 테스트 리팩토링(프로덕션 코드 변경 X)                        |
| setting          | 패키지 설치, 개발 설정                                                    |
| chore            | 빌드 테스트 업데이트, 패키지 매니저를 설정하는 경우(프로덕션 코드 변경 X) |
| rename           | 파일 혹은 폴더명을 수정하거나 옮기는 작업만인 경우                        |
| remove           | 파일을 삭제하는 작업만 수행한 경우                                        |


### 🤝 Commit Convention Detail
<div markdown="1">

- `<타입>`: `<제목> (<이슈번호>)` 의 형식으로 제목을 아래 공백줄에 작성
- 제목은 50자 이내 / 변경사항이 "무엇"인지 명확히 작성 / 끝에 마침표 금지
- 예) Feat: 로그인 기능 구현 (#5)


</div>

### 예시

```markdown
Feat: "추가 로그인 함수"  ---- 제목

로그인 API 개발           ---- 본문

Resolves: #123             ---- 꼬리말
Ref: #456
Related to: #48, #45
